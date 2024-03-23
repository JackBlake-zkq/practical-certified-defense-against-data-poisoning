import tensorflow as tf
from keras.models import Model
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Code Adapted From: https://github.com/ZuchniakK/MTKD/blob/main/distillation/train.py
arXiv:2302.07215
"""


AVAILABLE_MODES = ["output_avg", "loss_avg", "mimic_all", "supernet", "supernet2"]


def seq2func(seq_model):
    input_layer = tf.keras.layers.Input(batch_shape=seq_model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seq_model.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)

    func_model = tf.keras.models.Model([input_layer], [prev_layer])
    return func_model


class Distiller(Model):
    def __init__(self, student, teachers, mode="output_avg"):
        super(Distiller, self).__init__()
        self.teachers = teachers
        self.student = student
        self.mode = mode
        self.n_classes = self.student.layers[-1].output.get_shape()[-1]
        if self.mode not in AVAILABLE_MODES:
            raise NotImplementedError(f"mode: {self.mode} not implemented yet")

        if self.mode == "mimic_all":
            self.set_mimic_updates()

        self.teacher_aggregator = False
        if self.mode == "supernet":
            self.teacher_aggregator = True
            self.set_supernet_teacher()
        elif self.mode == "supernet2":
            self.teacher_aggregator = True
            self.set_supernet2_teacher()

    def set_mimic_updates(self):
        if isinstance(self.student.layers[-1], tf.keras.Model):
            last_layer = Model(
                inputs=self.student.layers[-1].input,
                outputs=self.student.layers[-1].get_layer("avg_pool").output,
            )
            self.student.pop()
            self.student.add(last_layer)

        else:
            self.student.pop()

        teacher_mimics = [
            tf.keras.layers.Dense(
                self.n_classes, activation="softmax", name=f"mimic_{i}"
            )(self.student.output)
            for i, teacher in enumerate(self.teachers)
        ]
        self.student = tf.keras.Model(inputs=self.student.input, outputs=teacher_mimics)

    def set_supernet_teacher(self):
        for i, teacher in enumerate(self.teachers):
            teacher.trainable = False
            teacher._name = teacher.name + f"_{i}"
            for l in teacher.layers:
                l._name = l.name + f"_{i}"
            teacher.layers[-1].layers[-1].activation = None

        data_input = tf.keras.layers.Input(shape=self.student.input.get_shape()[1:])
        teachers_layer = [t(data_input) for t in self.teachers]
        concatenate = tf.keras.layers.Concatenate(axis=-1)(teachers_layer)
        dense = tf.keras.layers.Dense(
            self.n_classes, activation="softmax", name=f"supernet_layer"
        )(concatenate)
        self.teachers = tf.keras.Model(inputs=data_input, outputs=dense)
        self.teachers.summary()

    def set_supernet2_teacher(self):
        for i, teacher in enumerate(self.teachers):
            teacher.trainable = False
            teacher._name = teacher.name + f"_{i}"
            for layer in teacher.layers:
                layer._name = layer.name + f"_{i}"
            if isinstance(teacher, tf.keras.Model):
                last_layer = Model(
                    inputs=teacher.layers[-1].input,
                    outputs=teacher.layers[-1].get_layer("avg_pool").output,
                )
                teacher.pop()
                teacher.add(last_layer)
            else:
                teacher.pop()

        data_input = tf.keras.layers.Input(shape=self.student.input.get_shape()[1:])
        teachers_layer = [t(data_input) for t in self.teachers]
        concatenate = tf.keras.layers.Concatenate(axis=-1)(teachers_layer)
        dense = tf.keras.layers.Dense(
            self.n_classes, activation="softmax", name=f"supernet_layer"
        )(concatenate)
        self.teachers = tf.keras.Model(inputs=data_input, outputs=dense)
        self.teachers.summary()

    def fit(self, x, validation_data, **kwargs):
        if self.teacher_aggregator:
            self.teachers.fit(
                x=x,
                validation_data=validation_data,
                epochs=50,
                steps_per_epoch=314,
                validation_steps=78,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        mode="min",
                        restore_best_weights=True,
                    )
                ],
            )
        super(Distiller, self).fit(x=x, validation_data=validation_data, **kwargs)

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        if self.teacher_aggregator:
            self.teachers.compile(
                optimizer=optimizer, metrics=metrics, loss=student_loss_fn
            )

    def call(self, inputs):
        return self.student(inputs)

    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass of teacher
            if self.mode in ["output_avg", "loss_avg", "mimic_all"]:
                teacher_predictions = [
                    teacher(x, training=False) for teacher in self.teachers
                ]
            elif self.mode in ["supernet", "supernet2"]:
                teacher_predictions = self.teachers(x, training=False)

            # Forward pass of student
            student_predictions = self.student(x, training=True)
            # Compute losses
            if self.mode == "output_avg":
                distillation_loss = self.distillation_loss_fn(
                    tf.reduce_mean(teacher_predictions, 0), student_predictions
                )
                student_loss = self.student_loss_fn(y, student_predictions)
            elif self.mode == "loss_avg":
                distillation_loss = sum(
                    [
                        self.distillation_loss_fn(
                            teacher_prediction, student_predictions
                        )
                        for teacher_prediction in teacher_predictions
                    ]
                ) / len(self.teachers)
                student_loss = self.student_loss_fn(y, student_predictions)
            elif self.mode == "mimic_all":
                distillation_loss = sum(
                    [
                        self.distillation_loss_fn(teacher_prediction, sp)
                        for teacher_prediction, sp in zip(
                            teacher_predictions, student_predictions
                        )
                    ]
                ) / len(self.teachers)
                student_predictions = tf.reduce_mean(student_predictions, 0)
                student_loss = self.student_loss_fn(y, student_predictions)
            elif self.mode in ["supernet", "supernet2"]:
                distillation_loss = self.distillation_loss_fn(
                    teacher_predictions,
                    student_predictions,
                )
                student_loss = self.student_loss_fn(y, student_predictions)

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)
        if self.mode == "mimic_all":
            y_prediction = tf.reduce_mean(y_prediction, 0)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


 # for i in range(self.n_subsets):
        #     if not os.path.exists(f'{self.state_dir}/base_models/model_{str(i)}.onnx'):
        #         print(f"Base model {i} not found. Aborting...")
        #         exit(1)

        # if not os.path.exists(f'{self.state_dir}/distillation'):
        #     os.mkdir(f'{self.state_dir}/distillation')

        # if not os.path.exists(f'{self.state_dir}/distillation/intermediates'):
        #     os.mkdir(f'{self.state_dir}/distillation/intermediates')

        # student.to("cpu")
        # student_untrained_onnx_path = f'{self.state_dir}/distillation/intermediates/student_untrained.onnx'
        # torch.onnx.export(
        #     student, 
        #     self.sample_input, 
        #     student_untrained_onnx_path, 
        #     opset_version=self.onnx_opset, 
        #     input_names=['input'], 
        #     output_names=['output']
        # )
        # student_untrained_keras_path = f'{self.state_dir}/distillation/intermediates/student_untrained.keras'
        # onnx2tf.convert(student_untrained_onnx_path, output_folder_path=student_untrained_keras_path , output_keras_v3=True, non_verbose=True)
        # tf_student = keras.saving.load_model(student_untrained_keras_path)
        # print("here")
        # teachers = []
        # for i in range(self.n_subsets):
        #     keras_base_model_path = f'{self.state_dir}/distillation/intermediates/model_{str(i)}.keras'
        #     onnx2tf.convert(f'{self.state_dir}/base_models/model_{str(i)}.onnx', output_folder_path=keras_base_model_path, output_keras_v3=True, non_verbose=True)
        #     teacher = keras.saving.load_model(keras_base_model_path)
        #     teachers.append(teacher)
        # model = Distiller(tf_student, teachers, mode="mimic_all")
        # model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=self.lr),
        #     metrics=["accuracy"],
        #     student_loss_fn=keras.losses.CategoricalCrossentropy(),
        #     distillation_loss_fn=keras.losses.KLDivergence()
        # )
        # callbacks = [
        #     keras.callbacks.ModelCheckpoint(
        #         filepath=f'{self.state_dir}/distillation/checkpoint',
        #         save_best_only=False,
        #         save_weights_only=False,
        #     ),
        # ]
        # imgs, labels = zip(*self.trainset)
        # model.fit(
        #     x=imgs,
        #     y=labels,
        #     epochs=50,
        #     callbacks=callbacks,
        #     validation_data=None,
        #     steps_per_epoch=314,
        # )
        # print(self.model.evaluate(x=self.testset, return_dict=True, steps=79))
        # onnx_model, _ = from_keras(model, self.sample_input, opset=self.onnx_opset)
        # onnx.save_model(onnx_model, f'{self.state_dir}/distillation/student.onnx')

    #         save_filename = f"{self.mode}_{self.architecture}_{self.fraction}_{self.n_teachers}_{self.uuid}"
    #         save_dir = os.path.join(
    #             MODELS_DIRECTORY, self.dataset, "students", save_filename
    #         )
    #         self.model.save(save_dir)

    #         csv_log_filename = os.path.join(MODELS_DIRECTORY, filename)
    #         fileEmpty = not os.path.isfile(csv_log_filename)
    #         with open(csv_log_filename, "a") as csvfile:
    #             headers = [
    #                 "model_location",
    #                 "dataset",
    #                 "architecture",
    #                 "n_teachers",
    #                 "teachers_fraction",
    #                 "mode",
    #                 "alpha",
    #                 "temperature",
    #                 "teachers_uuid",
    #                 "uuid",
    #                 "train_loss",
    #                 "train_accuracy",
    #                 "val_loss",
    #                 "val_accuracy",
    #                 "test_loss",
    #                 "test_accuracy",
    #             ]
    #             writer = csv.DictWriter(
    #                 csvfile, delimiter=",", lineterminator="\n", fieldnames=headers
    #             )
    #             if fileEmpty:
    #                 writer.writeheader()  # file doesn't exist yet, write a header

    #             writer.writerow(
    #                 {
    #                     "model_location": save_dir,
    #                     "dataset": self.dataset,
    #                     "architecture": self.architecture,
    #                     "n_teachers": self.n_teachers,
    #                     "teachers_fraction": str(self.fraction),
    #                     "mode": self.mode,
    #                     "alpha": self.alpha,
    #                     "temperature": self.temperature,
    #                     "teachers_uuid": ":".join(self.teachers_uuid),
    #                     "uuid": self.uuid,
    #                     "train_loss": train_loss,
    #                     "train_accuracy": train_accuracy,
    #                     "val_loss": val_loss,
    #                     "val_accuracy": val_accuracy,
    #                     "test_loss": test_loss,
    #                     "test_accuracy": test_accuracy,
    #                 }
    #             )


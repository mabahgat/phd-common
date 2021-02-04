from lime.lime_text import LimeTextExplainer
import tensorflow as tf


class Lime:

    def __init__(self, class_names):
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)
        self.num_features = 20
        self.num_samples = 20
        self.use_top_labels = True
        self.top_labels = 2
        self.investigate_labels = list(range(len(class_names)))

    def explain_text(self, text_str, predict_fn):
        """
        Explain the outcome from a text
        
        :param text_str: text to investigate
        
        :param predict_fn: lambda function to predict, should like

            def predict(raw_str_arr):
                encoded = tokenizer(raw_str_arr, truncation=True, padding=True)
                tf_slice = tf.data.Dataset.from_tensor_slices((dict(encoded), [0 for i in range(len(raw_str_arr))]))
                prob_result = model.predict(tf_slice.batch(1))[0]
                return prob_result

        :return: explaination object that can be used as follows:

            print(exp.as_list())
            exp.show_in_notebook()
            exp.as_pyplot_figure()

        """
        if self.use_top_labels:
            exp = self.explainer.explain_instance(text_str, predict_fn, num_features=self.num_features, num_samples=self.num_samples, top_labels=self.top_labels)
        else:
            exp = self.explainer.explain_instance(text_str, predict_fn, num_features=self.num_features, num_samples=self.num_samples, labels=self.investigate_labels)
        return exp

import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        n_samples, n_features = X.shape
        alpha = self.smoothing

        n_class_one = np.sum(y == 1)
        n_class_zero = np.sum(y == 0)

        self.class_one_prob_prior = np.log((n_class_one / n_samples) + 1e-6)
        self.class_zero_prob_prior = np.log((n_class_zero / n_samples) + 1e-6)

        feat_one_class_one_count = np.sum(X[y == 1], axis=0)
        feat_one_class_zero_count = np.sum(X[y == 0], axis=0)

        self.log_post_prob_feat_one_class_one = np.log((feat_one_class_one_count + alpha) / (n_class_one + 2 * alpha))
        self.log_post_prob_feat_one_class_zero = np.log((feat_one_class_zero_count + alpha) / (n_class_zero + 2 * alpha))

        feat_zero_class_one_count = n_class_one - feat_one_class_one_count
        feat_zero_class_zero_count = n_class_zero - feat_one_class_zero_count

        self.log_post_prob_feat_zero_class_one = np.log((feat_zero_class_one_count + alpha) / (n_class_one + 2 * alpha))
        self.log_post_prob_feat_zero_class_zero = np.log((feat_zero_class_zero_count + alpha) / (n_class_zero + 2 * alpha))

        return

    def predict(self, X):
        one_class_prob = np.sum((X * self.log_post_prob_feat_one_class_one + (1 - X) * self.log_post_prob_feat_zero_class_one), axis=1) + self.class_one_prob_prior

        zero_class_prob = np.sum((X * self.log_post_prob_feat_one_class_zero + (1 - X) * self.log_post_prob_feat_zero_class_zero), axis=1) + self.class_zero_prob_prior


        return (one_class_prob > zero_class_prob).astype(int)

































        
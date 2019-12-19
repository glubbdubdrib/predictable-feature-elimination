import unittest


class TestPFE(unittest.TestCase):

    def test_PFE(self):

        from sklearn.datasets import make_friedman1
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVR
        from pfe.pfe import PFE

        X, y = make_friedman1(n_samples=50, n_features=1000, random_state=0)
        estimator = SVR(kernel="linear")

        selector = RFE(estimator, 5, step=1)
        selector = selector.fit(X, y)
        print(sum(selector.support_))

        selector = PFE(estimator)
        selector = selector.fit(X, y)
        print(sum(selector.support_))

    def test_dataset(self):
        from sklearn.datasets import load_wine
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import RidgeClassifier
        from sklearn import clone
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import Pipeline
        from pfe.pfe import PFE

        seed = 42

        X, y = load_wine(return_X_y=True)

        clf = RidgeClassifier(random_state=seed)

        pipeline_rfe = Pipeline([("rfe", RFE(clone(clf))), ("clf", clone(clf))])
        pipeline_pfe = Pipeline([("pfe", PFE(clone(clf))), ("clf", clone(clf))])

        pipeline_rfe.fit(X, y)
        score_rfe = pipeline_rfe.score(X, y)
        n_features_rfe = sum(pipeline_rfe.steps[0][1].support_)

        pipeline_pfe.fit(X, y)
        score_pfe = pipeline_pfe.score(X, y)
        n_features_pfe = sum(pipeline_pfe.steps[0][1].support_)

        print()
        print("RFE score: %.4f; #features: %d" % (score_rfe, n_features_rfe))
        print("PFE score: %.4f; #features: %d" % (score_pfe, n_features_pfe))


suite = unittest.TestLoader().loadTestsFromTestCase(TestPFE)
unittest.TextTestRunner(verbosity=2).run(suite)

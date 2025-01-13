from sklearn.model_selection import GridSearchCV, train_test_split

from skfolio import Population, RatioMeasure
from skfolio.distance import KendallDistance, PearsonDistance
from skfolio.cluster import LinkageMethod

from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
    optimal_folds_number,
)

from skfolio.optimization import EqualWeighted

from skfolio.metrics import make_scorer

from skfolio.preprocessing import prices_to_returns

class Comparator:
    def __init__(self, prices, models, log_returns=False, test_size=0.2):

        self.prices = prices
        self.X = prices_to_returns(prices, log_returns=log_returns)
        self.X_train, self.X_test = train_test_split(self.X, test_size=test_size, shuffle=False)

        self.cv = WalkForward(train_size=252, test_size=60)

        self.searchers = []
        for model in models:
            grid_searcher = GridSearchCV(
                estimator=model,
                cv=self.cv,
                n_jobs=-1,
                param_grid={
                    "distance_estimator": [PearsonDistance(), KendallDistance()],
                    "hierarchical_clustering_estimator__linkage_method": [
                        LinkageMethod.SINGLE,
                        LinkageMethod.WARD,
                        LinkageMethod.COMPLETE,
                    ],
                },
                scoring=make_scorer(RatioMeasure.CVAR_RATIO),
            )
            self.searchers.append(grid_searcher)

        self.models = models
        self.population = Population([])

        self.benchmark = EqualWeighted()
        self.best_models = []


    def run(self):
        
        self.population = Population([])
        self.best_models = []
        
        best_preds = []
        for searcher in self.searchers:
            searcher.fit(self.X_train)
            model = searcher.best_estimator_

            self.best_models.append(model)
            preds = cross_val_predict(
                model,
                self.X_test,
                cv=self.cv,
                n_jobs=-1,
                portfolio_params=model.portfolio_params,
            )    
            self.population.append(preds)

        model = self.benchmark
        model.fit(self.X_train)
        preds = cross_val_predict(
                model,
                self.X_test,
                cv=self.cv,
                n_jobs=-1,
                portfolio_params=dict(name="Equal Weight Benchmark"),
            ) 
        self.population.append(preds)
        return 
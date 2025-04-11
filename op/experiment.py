class Experiment:
    """
        A experiment should recieve everything which is important to a run.

        The idea is to standardize the experiment, so you can either run a prod or dev run:
            python experiment.py run
        or 
            python experiment.py run-development 

        It should: 
            - be able to run the steps of the experiment.
            - be able to save the results of the experiment.
    """
    def __init__(self, epochs: int = None, dataset: str = None):
        self.epochs = epochs
        self.dataset = dataset
        pass

    def __validate_instantiated(self):
        attributes = [
            'epochs', 
            'dataset', 
            'preprocessors'
        ]
        for attr in attributes:
            if getattr(self, attr, None) is None:
                raise ValueError(f"The attribute '{attr}' must be initialized and cannot be None.")
    
    def train(self):
        # To be defined
        pass

    def run(self):
        # Run the steps of the experiment
        #     What composes?

        # Steps
        # - Standardized sources
        # - Analytics of sources
        # - Preproc
        # - Analytics of input data
        # - Train
        # - Test
        # - Results
        pass

    def run_development(self):
        # Parameters
        epochs = 1
        dataset = self.dataset

        
        


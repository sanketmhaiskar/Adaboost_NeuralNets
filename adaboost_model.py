class Adaboost_model:
    def __init__(self):
        self.score = 0.0
        self.correct_classified_list = set()
        self.alpha = 0.0
        self.first_index = 0
        self.second_index = 0
# stump contains detail of one stump with 4 classifiers
class Stump:
    def __init__(self):
        self.first_index = 0
        self.second_index = 0
        self.score_0 = 0.0
        self.score_90 = 0.0
        self.score_180 = 0.0
        self.score_270 = 0.0
        self.correct_classified_0 = set()
        self.correct_classified_90 = set()
        self.correct_classified_180 = set()
        self.correct_classified_270 = set()

#best stump contains the detail for the best stump
class Best_Stump:
    def __init__(self):
        self.score = 0.0
        self.correct_classified_list = set()
        self.alpha = 0.0
        self.first_index = 0
        self.second_index = 0
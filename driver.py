import nnet


def driver():
    nnet.main("train-data.txt", "test-data.txt", 20)


driver()

import os
import json
import pytest

from predict import Predictor

TEST_DIR = "tests"

ARGS_FILE = os.path.join(TEST_DIR,"default_args.json")

ARGS = {}
with open(ARGS_FILE) as f:
    ARGS = json.load(f)

SST_FILE = os.path.join(TEST_DIR,"default_sst_input.txt")
SST_EVAL_FILE = os.path.join(TEST_DIR,"default_sst_eval.txt")

class Test_predictor_SST:

    @pytest.fixture(scope='class')
    def predictor(self) -> Predictor:
        return Predictor(
            ARGS["checkpoint_dir"],
            "sst",
            ARGS["batch_size"],
            ARGS["cuda"]
            )

    def test_predict_from_file(self,predictor):

        output = predictor.predict_from_file(SST_FILE)
        assert len(output[0])==3
        assert output[0] == [0,0,1]

        assert len(output[1])==3

    def test_predict_from_file_eval(self,predictor):
        output = predictor.predict_from_file(SST_EVAL_FILE,
                                        evaluate=True)
        assert len(output[0])==3
        assert output[0] == [0,0,1]

        assert len(output[1])==3

        assert isinstance(output[2],dict)
        assert len(output[2])>0
        assert output[2]["ACC"]==100

    def test_predict_from_texts(self,predictor):
        texts = ["It is very bad",
                  "The worst kind of service",
                  "I am happy"]
        output = predictor.predict([[t] for t in texts])
        assert len(output[0])==3
        assert output[0] == [0,0,1]

        assert len(output[1])==3

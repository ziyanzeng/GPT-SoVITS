import re

re_numbers = re.compile(r"[0-9\.]+")

class Predict:
    def __init__(self, params: str) -> None:
        self.models = self.__load_model_params(params)

    def _arrange_order(self, text: str, keep_words: list, stop_words: list):
        start_indexs = {}
        order_words = []
        text_buf = bytearray(text.encode())

        sorted_words = []
        for t,words in enumerate([keep_words,stop_words]):
            for w in words:
                sorted_words.append({'word':w,'type':t})
        
        sorted_words = sorted(sorted_words, key=lambda item: -len(item["word"]))
        for item in sorted_words:
            wbuf = item["word"].encode()
            last_index = 0 if item["word"] not in start_indexs else start_indexs[item["word"]]
            next_index = text_buf.find(wbuf,last_index)
            if next_index >= 0:
                text_buf[next_index:next_index+len(wbuf)] = (' '*len(wbuf)).encode()
                next_index = next_index+len(wbuf)
            start_indexs[item["word"]] = next_index
            order_words.append({"item": item, "start": next_index})
        order_words = sorted(order_words, key=lambda x: x["start"])

        split_words = [[],[]]
        for v in order_words:
            item = v["item"]
            split_words[item["type"]].append(item["word"])
        
        return split_words[0],split_words[1]
    
    def do_predict(self, text: str, proba: str) -> dict:
        return NotImplemented

    def predict(self, text: str, proba: str) -> dict:
        predict_result = self.do_predict(text,proba)
        predict_result[0]["keep_cut_words"][0],predict_result[0]["stop_word_in_sentences"][0] = self._arrange_order(
            text, predict_result[0]["keep_cut_words"][0],predict_result[0]["stop_word_in_sentences"][0]
        )
        return predict_result

    def predict_old(self, text: str, proba: str) -> dict:
        return self.predict(text, proba)

    def predict_new(self, text, proba) -> dict:
        proba = "90"
        predict_result = self.predict(
            text,
            proba,
        )

        keep_cut_words = []
        if predict_result and len(predict_result) >= 3:
            word_map = {}
            for word in predict_result[0]["keep_cut_words"][0]:
                word_map[word] = 1 if word not in word_map else word_map[word] + 1

            vocab_dict = predict_result[0]["vocab_dict"]
            for word, num in word_map.items():
                keep_cut_words.append(
                    {
                        "word": word,
                        "tfidf": round(
                            predict_result[1].tolist()[0][vocab_dict[word]], 4
                        ),
                        "num": num,
                        "frequency": round(
                            num
                            / (
                                len(predict_result[0]["keep_cut_words"][0])
                                + len(predict_result[0]["stop_word_in_sentences"][0])
                            ),
                            4,
                        ),
                    }
                )
        return {"keepCutWords": keep_cut_words, "predict": predict_result[2]}

    def __load_model_params(self, params: str) -> dict:
        models = []
        params = params.replace("%", "")
        lines = params.split("\n")[1:]
        for line in lines:
            cells = line.split("\t")

            numbers = re_numbers.findall(cells[6])
            models.append(
                {
                    "id": cells[0],
                    "precision": float(cells[1]),
                    "recall": float(cells[2]),
                    "accuracy": float(cells[3]),
                    "f1score": float(cells[4]),
                    "learningRate": float(numbers[0]),
                    "batchSize": int(numbers[1]),
                    "epoch": int(numbers[2]),
                    "maxFeatures": int(numbers[3]),
                }
            )
        return sorted(models, key=lambda x: -int(x["id"]))

    def match_model(
        self, learningRate: float, epoch: int, batchSize: int, maxFeatures: int
    ) -> str:
        for m in self.models:
            if m["epoch"] <= epoch and m["maxFeatures"] >= maxFeatures:
                return m["id"]
        return self.models[len(self.models) - 1]["id"]

    def get_model_info(self, id: str):
        for m in self.models:
            if m["id"] == id:
                return {
                    "precision": str(m["precision"]),
                    "recall": str(m["recall"]),
                    "accuracy": str(m["accuracy"]),
                    "f1score": str(m["f1score"]),
                }
        return None


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from python_packages import config

    conf = config.get_config()
    testParams = conf["predict"]["modelParams"]

    h = Predict(testParams)
    print(testParams)
    model_id = h.match_model(
        learningRate=0.01, epoch=30, batchSize=128, maxFeatures=1000
    )
    print(model_id, h.get_model_info(model_id))

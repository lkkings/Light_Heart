from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

_config = {
    "user": "root",
    "password": "",
    "db": "default",
    "host": "localhost",
    "port": "19530"
}


class SignMapper:
    def __init__(self):
        self.collection: Collection | None = None

    def connect(self):
        connections.connect(_config["db"], host=_config["host"], port=_config["port"])
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="sign",
                dtype=DataType.FLOAT_VECTOR,
                dim=142
            ),
            FieldSchema(
                name="prompt",
                dtype=DataType.VARCHAR,
                max_length=64
            )
        ]
        schema = CollectionSchema(fields, "用于测试")
        self.collection = Collection("algorithm_b_2", schema)
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.collection.create_index("sign", index)
        self.collection.load()

    def insert(self,sign_id: list, prompt: str):
        data = [[sign_id], [prompt]]
        self.collection.insert(data)
        self.collection.flush()

    def search_by_id(self, sign: list, n=1):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": n},
        }
        return self.collection.search([sign], "sign", search_params, limit=n, output_fields=["prompt"])

    def close(self):
        self.collection.release()
        connections.disconnect("default")

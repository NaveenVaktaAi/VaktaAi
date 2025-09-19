from app.config.settings import Settings
from pymilvus import connections




milvus_alias = "default"



def connect_to_milvus():
    global milvus_alias
    print("==================================================")
    connections.connect(
        alias=milvus_alias,
        host=Settings.MILVUS_HOST,
        port=Settings.MILVUS_PORT,
    )

    print("-------------------done-------------------")


def disconnect_from_milvus():
    global milvus_alias
    connections.disconnect(alias=milvus_alias)

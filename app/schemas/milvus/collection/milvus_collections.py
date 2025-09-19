import logging
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
from pymilvus.exceptions import ConnectionNotExistException, MilvusException
from app.schemas.milvus.client import connect_to_milvus

__all__ = (
    "chunk_msmarcos_collection",
)

logger = logging.getLogger(__name__)

id = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="Id",
)
vector = FieldSchema(
    name="vector",
    dtype=DataType.FLOAT_VECTOR,
    dim=768,
)
mongo_chunk_id = FieldSchema(
    name="mongo_chunk_id",
    dtype=DataType.VARCHAR,
    max_length=1024,
    description="chunk text",
)
mongo_document_id = FieldSchema(
    name="mongo_document_id",
    dtype=DataType.VARCHAR,
    max_length=1024,
    description="Document Id",
)

def create_collection(model: str, name: str, tries: int = 0) -> Collection | None:
    collection_name = f"{name}_{model}"
    fields = [id, vector, mongo_document_id]
    
    if name == "chunk":
        fields = [id, mongo_chunk_id, vector, mongo_document_id]
    
    schema = CollectionSchema(
        fields=fields,
        description=f"{name} search for {model} embeddings",
    )

    try:
        # Check if collection exists
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            existing_collection = Collection(collection_name)
            existing_schema = existing_collection.schema
            new_schema_dict = {f.name: f for f in schema.fields}
            existing_schema_dict = {f.name: f for f in existing_schema.fields}
            
            if new_schema_dict.keys() != existing_schema_dict.keys() or \
               any(new_schema_dict[f].dtype != existing_schema_dict[f].dtype or 
                   new_schema_dict[f].params != existing_schema_dict[f].params for f in new_schema_dict):
                logger.warning(f"Schema mismatch for {collection_name}. Dropping and recreating.")
                utility.drop_collection(collection_name)
            else:
                if not existing_collection.has_index():
                    existing_collection.create_index(
                        field_name="vector",
                        index_params={
                            "metric_type": "IP",
                            "index_type": "IVF_FLAT",
                            "params": {"nlist": 1024},
                        },
                    )
                existing_collection.load()
                logger.info(f"Reusing existing collection {collection_name}")
                return existing_collection

        # Create new collection
        kwargs = {
            "name": collection_name,
            "schema": schema,
            "using": "default",
            "shards_num": 2,
        }
        collection = Collection(**kwargs)
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            },
        )
        collection.load()
        logger.info(f"Collection {collection_name} created successfully")
        return collection

    except ConnectionNotExistException:
        try:
            connect_to_milvus()
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            if tries < 5:
                return create_collection(model, name, tries + 1)
            raise e
        return create_collection(model, name, tries + 1)
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        raise

chunk_msmarcos_collection = create_collection(
    "msmarco_distilbert_base_tas_b", "chunk"
)
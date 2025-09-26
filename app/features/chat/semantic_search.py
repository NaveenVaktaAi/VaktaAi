import asyncio
import logging
from typing import TYPE_CHECKING

from asgiref.sync import sync_to_async
from sentry_sdk import capture_exception
from app.common import constants


from pymilvus import MilvusException, SearchResult
from app.utils.transformers.models import msmarco_model
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pymilvus import Collection
    from sentence_transformers import SentenceTransformer


class ResponseNotFound(Exception):
    pass


class BotGraphMixin:
    def __init__(
        self,
        *,
        db: Session,
    ):
        self.db = db

    # async def _get_answers(self, filtered_ids: list[str]):
    #     try:
    #         texts = (
    #             self.db.query(QuestionAnswer.answer)
    #             .filter(QuestionAnswer.id.in_(filtered_ids))
    #             .all()
    #         )
    #     except Exception as e:
    #         capture_exception(e)
    #         return None
    #     else:
    #         return [text for (text,) in texts]

    async def _get_milvus_response(
        self,
        collection: "Collection",
        model: "SentenceTransformer",
        questions: str,
        mongo_document_id: int = None,
    ):
        try:
            encoded_embedding = model.encode(
                questions, normalize_embeddings=True
            ).tolist()

            search_params = {"metric_type": "IP"}
            filter_condition = None
            if mongo_document_id:
                # If org_id is provided, use it in the filter
                filter_condition = f"mongo_document_id == {mongo_document_id}"

            response: SearchResult = await sync_to_async(collection.search)(
                [encoded_embedding],
                "vector",
                search_params,
                limit=constants.MILVUS_TOP_K_LIMIT,
                output_fields=["id", "mongo_chunk_id"],
                expr=filter_condition,
            )
            print("response>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>_get_milvus_response>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",response)
        except MilvusException as e:
            return None

        else:
            return response
        
        
        
    async def _get_milvus_answer_response(
        self,
        collection: "Collection",
        model: "SentenceTransformer",
        questions: str,
        mongo_document_id: str = None,
    ):
        try:
            print("questions>>>>>>>>in milvus>>>>>>>>>>>>>>",questions)
            encoded_embedding = model.encode(
                questions, normalize_embeddings=True
            ).tolist()

            search_params = {"metric_type": "IP"}
            filter_condition = None
            if mongo_document_id:
                filter_condition = f"mongo_document_id == {mongo_document_id}"

            response: SearchResult = await sync_to_async(collection.search)(
                [encoded_embedding],
                "vector",
                search_params,
                limit=constants.MILVUS_TOP_K_LIMIT,
                output_fields=["id","mongo_chunk_id"],
                expr=filter_condition,
            )
            print("response>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>_get_milvus_response>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",response)
        except MilvusException as e:
            return None

        else:
            return response

    def _get_milvus_id(
        self,
        collection: "Collection",
        model: "SentenceTransformer",
        questions: list[str],
        messageId: str,
    ):
        try:
            encoded_embedding = model.encode(
                " ".join(questions), normalize_embeddings=True
            ).tolist()
            search_params = {"metric_type": "IP"}
            response: SearchResult = collection.search(
                [encoded_embedding],
                "vector",
                search_params,
                limit=1,
                expr=f"mongo_chunk_id == {messageId}",
                output_fields=[
                    "id",
                    "mongo_chunk_id",
                ],
            )

        except MilvusException as e:
            return None

        else:
            return response

    # async def _search_questions_vector(
    #     self,
    #     collection: "Collection",
    #     model: "SentenceTransformer",
    #     questions: list[str],
    #     mongo_document_id: int,
    # ) -> tuple[list[str], list[str], list[str]] | None:
    #     answer_ids = []
    #     for question in questions:
    #         response_long: SearchResult = await self._get_milvus_response(
    #             collection, model, question, mongo_document_id
    #         )

    #         new_ids = await sync_to_async(self.find_filtered_ids)(
    #             response_long,
    #             (
    #                 constants.MILVUS_DISTANCE_LIMIT_FOR_QUESTIONS
    #                 if (collection == questions_msmarcos_collection)
    #                 else constants.MILVUS_DISTANCE_LIMIT
    #             ),
    #         )

    #         answer_ids = answer_ids + new_ids

    #     if len(answer_ids) > 0:
    #         return list(set(answer_ids))
    #     return None
    
    async def _search_answers_vector(
        self,
        collection: "Collection",
        model: "SentenceTransformer",
        questions: list[str],
        mongo_document_id: int,
    ) -> tuple[list[str], list[float], list[str]] | None:
        """
        Searches for relevant answers, returning answer IDs, distances, and answer vectors.

        :param collection: Milvus collection to query
        :param model: SentenceTransformer model for embeddings
        :param questions: List of user questions
        :param mongo_document_id: MongoDB Document ID for contextual filtering
        :return: Tuple containing lists of answer IDs, distances, and vectors, or None if no results found
        """
        print("questions>>>>>>>>>>>>>>>>>>>>>>",questions)
        for question in questions:
            response_long = await self._get_milvus_answer_response(
                collection, model, question, mongo_document_id
            )
            
            print("response_long:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", response_long)
            
            filtered_answers = await sync_to_async(self.find_filtered_answer)(
                response_long
            )
            
            print("answer_ids:>>>>>>>>>>>>>>>>>>>>>>>>>_search_answers_vector>>>>>>>>>>>>>>>>>", filtered_answers)            
 

        if filtered_answers:
            
            return list(set(filtered_answers))
        return None

    def find_filtered_ids(self, response, distance_limit):
        filtered_answer_ids = []
        if len(response) and len(response[0].ids):
            print(response, "response response response")
            for i, distance in enumerate(response[0].distances):
                if distance > distance_limit:
                    id = response[0][i].entity.get("id")
                    if id in response[0].ids:
                        filtered_answer_ids.append(response[0][i].entity.get("mongo_chunk_id"))
        return filtered_answer_ids
    
    
    def find_filtered_answer(self, response):
        contexts = []
        
        if len(response) and len(response[0]):
            for i, distance in enumerate(response[0].distances):
                entity = response[0][i].entity
                if entity.get("id") in response[0].ids:
                    contexts.append(
                         entity.get("mongo_chunk_id"),
                    )
        return contexts

   


    
    async def search_answers(
        self, questions: list[str], collection: "Collection", mongo_document_id: int
    ) -> list[str] | None:
        """
        Searches for relevant questions to respond to the user.
        """
        print("questions type:", type(questions))
        print("questions content:", questions)

        if not questions:
            logger.warning("No questions provided for search.")
            return None

        # Convert dict questions to strings
        questions = [str(q) if isinstance(q, dict) else q for q in questions]

        try:
            msmarco_results = await self._search_answers_vector(
                collection, msmarco_model, questions, mongo_document_id
            )
            print("msmarco_results type:", type(msmarco_results))
            print("msmarco_results content:", msmarco_results)

            return msmarco_results if msmarco_results else None

        except Exception as e:
            logger.error(f"Error in search_questions: {e}")
            capture_exception(e)
            return None

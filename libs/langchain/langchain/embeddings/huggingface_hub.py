from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env
import math
import os
import copy
import requests
import time

DEFAULT_REPO_ID = "sentence-transformers/all-mpnet-base-v2"
VALID_TASKS = ("feature-extraction",)


# DEFAULT_API_URL = ""
# DEFAULT_HEADERS = {
# 	"Authorization":"Bearer hf_11",
# 	"Content-Type": "application/json"
# }


class HuggingFaceHubEmbeddings(BaseModel, Embeddings):
    """HuggingFaceHub embedding models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceHubEmbeddings
            repo_id = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceHubEmbeddings(
                repo_id=repo_id,
                task="feature-extraction",
                huggingfacehub_api_token="my-api-key",
            )
    """

    client: Any  #: :meta private:
    repo_id: str = DEFAULT_REPO_ID
    """Model name to use."""
    task: Optional[str] = "feature-extraction"
    """Task to call the model with."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""

    huggingfacehub_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )
        try:
            from huggingface_hub.inference_api import InferenceApi

            repo_id = values["repo_id"]
            if not repo_id.startswith(("sentence-transformers", "intfloat", "BAAI", "hkunlp")):
                raise ValueError(
                    "Currently only 'sentence-transformers', 'intfloat', 'BAAI', 'hkunlp' embedding models "
                    f"are supported. Got invalid 'repo_id' {repo_id}."
                )
            client = InferenceApi(
                repo_id=repo_id,
                token=huggingfacehub_api_token,
                task=values.get("task"),
            )
            if client.task not in VALID_TASKS:
                raise ValueError(
                    f"Got invalid task {client.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
            values["client"] = client
        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = self.client(inputs=texts, params=_model_kwargs)
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response

MAX_BATCH_SIZE_FOR_API = 32
class HuggingFaceAPIEmbeddings(BaseModel, Embeddings):
    """HuggingFaceAPI embedding models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceHubEmbeddings
            repo_id = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceHubEmbeddings(
                repo_id=repo_id,
                task="feature-extraction",
                huggingfacehub_api_token="my-api-key",
            )
    """

    client: Any  #: :meta private:
    repo_id: str = DEFAULT_REPO_ID
    """Model name to use."""
    task: Optional[str] = "feature-extraction"
    """Task to call the model with."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""    
    api_url: str
    headers: dict
    max_batch_size: int = MAX_BATCH_SIZE_FOR_API
    

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""        
        try:            
            import requests 
            # if not repo_id.startswith("sentence-transformers"):
            #     raise ValueError(
            #         "Currently only 'sentence-transformers' embedding models "
            #         f"are supported. Got invalid 'repo_id' {repo_id}."
            #     )
            client = requests            
            values["client"] = client
        except ImportError:
            raise ImportError(
                "Could not import requests python package. "
                "Please install it with `pip install requests`."
            )
        return values

    def wait_until_ready(self):
        '''waits until is model is ready to serve'''
        wait_time = 0
        while True:
            if self.health_check():
                break
            else:
                # logger.info(f'waiting for model to be ready...{wait_time} secs')
                time.sleep(5) 
                wait_time += 5
    
    def health_check(self):
        _model_kwargs = self.model_kwargs or {}
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": 'check model health', "parameters": _model_kwargs})
        if response.status_code == 200:
            return True
        else:
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        self.wait_until_ready()
        
        responses = []
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        req_size = len(texts)
        if req_size>self.max_batch_size:
            #create requests in batch
            no_of_batches = math.ceil(req_size/self.max_batch_size)
            for bt_id in range(no_of_batches):
                if req_size>=(bt_id+1)*(self.max_batch_size):
                    end_idx =  (bt_id+1)*(self.max_batch_size)
                else:
                    end_idx =  req_size
                batch_texts = texts[bt_id*(self.max_batch_size):end_idx]         
                batch_responses = requests.post(self.api_url, headers=self.headers, json={"inputs": batch_texts})
                batch_responses = batch_responses.json()
                if self.repo_id == 'aws-bert-base-mdoc-bm25-9243':
                    batch_responses_pre = copy.deepcopy(batch_responses)
                    batch_responses = batch_responses['embeddings']
                responses = responses + batch_responses
        else:
            responses = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "parameters": _model_kwargs})
            responses = responses.json()
            if self.repo_id == 'aws-bert-base-mdoc-bm25-9243':
                    responses_pre = copy.deepcopy(responses)
                    responses = responses['embeddings']
            # responses = [res.json() for res in responses]
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response



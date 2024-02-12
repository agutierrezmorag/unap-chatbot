from typing import List, Optional

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class CustomWikipediaRetriever(WikipediaRetriever):
    page_name: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.page_name:
            query = self.page_name
        return self.load(query=query)

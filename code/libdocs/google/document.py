import re
from typing import List

from google.cloud import documentai


class GooglePDFProcessor:
    """
    GooglePDFProcessor is capable of reading docs from GCS buckets. It is capable of
    processing these using Document AI on files to provide the contents.
    """

    def __init__(
        self,
        project_id="development-398309",
        location="us",
        processor_id="277f11647ef22bec",
    ):
        """
        Construct a new 'GooglePDFProcessor' object.

        :param project: The project name for GCS.
        :param gcs: The GoogleCloudStorage object used to access the documents.
        :return: returns nothing
        """
        if project_id is None or project_id == "":
            raise Exception("project_id is not set or empty")
        if location is None or location == "":
            raise Exception("location is not set or empty")
        if processor_id is None or processor_id == "":
            raise Exception("processor_id is not set or empty")
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_path = self.client.processor_path(
            self.project_id, self.location, self.processor_id
        )

    def _layout_to_text(
        self, layout: documentai.Document.Page.Layout, text: str
    ) -> str:
        """
        Given a layout return the actual text.

        :param layout: The document layout.
        :param text: The document text.
        :return: returns actual text.
        """
        # If a text segment spans several lines, it will
        # be stored in different text segments.
        return "".join(
            text[int(segment.start_index) : int(segment.end_index)]
            for segment in layout.text_anchor.text_segments
        )

    def _remove_numbers(self, text: str) -> str:
        # Remove lines that are numbers or number.number
        text = re.sub(r"^[0-9\.\s]+$", " ", text)
        return text

    def _pages_to_content(self, pages, text) -> List[str]:
        """
        Provide the files in the bucket.

        :return: returns list of files.
        """
        # Extract text from the document
        content = []
        for page in pages:
            for paragraph in page.paragraphs:
                paragraph_text = self._layout_to_text(paragraph.layout, text)
                paragraph_text = self._remove_numbers(paragraph_text)
                if paragraph_text.strip():
                    content.append(paragraph_text)
        return content

    def contents_for_bytes(self, data: bytes) -> List[str]:
        """
        Provides the contents of a PDF file.

        :param data: PDF file as bytes
        :return: returns list of paragraphs.
        """

        raw_document = documentai.RawDocument(
            content=data, mime_type="application/pdf"
        )
        request = documentai.ProcessRequest(
            name=self.processor_path, raw_document=raw_document
        )

        # Process the document
        result = self.client.process_document(request=request)

        # Extract text from the document
        return self._pages_to_content(
            result.document.pages, result.document.text
        )

    def contents_for_gcs_uri(self, gcs_uri: str) -> List[str]:
        """
        Provides the contents of a PDF file.

        :param gcs_uri: GCS URI to the PDF file
        :return: returns list of paragraphs
        """

        gcs_document = documentai.GcsDocument(
            gcs_uri=gcs_uri, mime_type="application/pdf"
        )
        request = documentai.ProcessRequest(
            name=self.processor_path, gcs_document=gcs_document
        )

        # Process the document
        result = self.client.process_document(request=request)

        # Extract text from the document
        return self._pages_to_content(
            result.document.pages, result.document.text
        )

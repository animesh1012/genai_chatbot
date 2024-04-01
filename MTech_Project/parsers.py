from langchain.schema import BaseOutputParser
import re

class OutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        """Cleans output text"""
        text = text.strip()
        # remove suffix containing words from stop_words list
        stopwords=['Human:', 'AI:']
        for word in stopwords:
            text = text.removesuffix(word)
        # text = re.findall('.*(?=Human:)',text,re.DOTALL)[0]
        # text = re.findall('.*(?=AI:)',text,re.DOTALL)[0]
        text = text.split('Human: ')[0]
        text = text.split('AI: ')[0]
        return text.strip()

    @property
    def _type(self) -> str:
        """Return output parser type for serialization"""
        return "output_parser"


from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import pandas as pd

import logging
import os

logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {lineno} - {message}", style='{')
logger = logging.getLogger("PandasAI")

logger.info("Starting PandasAI example")
try :
    logger.info("Setting up llm and data")
    llm = OpenAI(api_token = os.getenv("OPENAI_API_KEY", default=None))
    answering_machine = PandasAI(llm, enforce_privacy=True, conversational=False)
    nz_parliament_dataframe = pd.read_csv("NZ_Parliament.csv")

    question = "Who is the MP for Auckland central?"
    logger.info(f"{question} : ")
    answer = answering_machine.run(nz_parliament_dataframe, prompt=question)
    logger.info(f"{answer}")

    question = "How many MPs are Green ?"
    logger.info(f"{question} : ")
    answer = answering_machine.run(nz_parliament_dataframe, prompt=question)
    logger.info(f"{answer}")

    question = "Plot the histogram of Parties showing for each total of Names , using different colours for each Party"
    logger.info(f"{question} : ")
    answer = answering_machine.run(nz_parliament_dataframe, prompt=question)

except :
    logger.exception(f"Error received : ")

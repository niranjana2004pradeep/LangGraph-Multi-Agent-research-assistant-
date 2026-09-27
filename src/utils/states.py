from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from utils.objects import Analyst
from langgraph.graph import MessagesState
import operator

#state
class GenerateAnalystsState(TypedDict):
    topic: str #Research topic
    max_analysts: int # number of analysts to generate
    human_analyst_feedback: NotRequired[Optional[str]] # whether to include human analyst feedback
    analysts: NotRequired[List[Analyst]] # list of all our analysts

class InterviewState(MessagesState):
    max_num_turns: int # number turns of conversation
    context: Annotated[List, operator.add] # source of docs
    analyst: Analyst # my analyst
    interview: str # interview transcript
    sections: list # final key we duplicate in outer state for Send() api

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
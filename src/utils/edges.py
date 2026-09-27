from dotenv import load_dotenv
from utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState
from typing import Literal
from langgraph.graph import END
from langchain.messages import AIMessage
from langgraph.types import Send
from langchain.messages import HumanMessage


load_dotenv()

#conditional edges
def should_continue(state: GenerateAnalystsState)-> Literal["create_analysts", "__end__"]:
    """Return the next node to execute"""

    human_analyst_feedback = state.get("human_analyst_feedback","")

    if human_analyst_feedback:
        return "create_analysts"

    return END

def routes_messages(state: InterviewState, name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    
    return "ask_question"       

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback')
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?"
                                           )
                                                       ]}) for analyst in state["analysts"]]
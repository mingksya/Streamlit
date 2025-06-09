import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


MODEL_PRICES = {
    "input": {
        "gpt-3.5-turbo": 0.5 / 1_000_000,
        "gpt-4o": 5 / 1_000_000,
        "claude-3-5-sonnet-20240620": 3 / 1_000_000,
        "gemini-1.5-pro-latest": 3.5 / 1_000_000
    },
    "output": {
        "gpt-3.5-turbo": 1.5 / 1_000_000,
        "gpt-4o": 15 / 1_000_000,
        "claude-3-5-sonnet-20240620": 15 / 1_000_000,
        "gemini-1.5-pro-latest": 10.5 / 1_000_000
    }
}


def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
	# # clear_button이 클릭되거나 message_history에 이럭이 없는 경우에 초기화
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]


def select_model():
    # 슬라이더를 추가하고 temperature를 0부터 2까지 선택할 수 있도록 한다
	# 초깃값은 0.0, 눈금은 0.01로 한다
	
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
        return ChatOpenAI(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "GPT-4":
        st.session_state.model_name = "gpt-4o"
        return ChatOpenAI(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Claude 3.5 Sonnet":
        st.session_state.model_name = "claude-3-5-sonnet-20240620"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Gemini 1.5 Pro":
        st.session_state.model_name = "gemini-1.5-pro-latest"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name
        )


def init_chain():
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
		("user", "{user_input}")  # 이 곳에 나중에 사용자의 입력이 들어간다
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser


def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        # Claude 3는 토크나이저를 공개하지 않아서, tiktoken을 사용해 토큰 수를 계산한다
        # 정확한 토큰 수는 아니지만, 대략적인 토큰 수를 파악할 수 있다
        if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 임시로 이용
        return len(encoding.encode(text))


def calc_and_display_costs():
    output_count = 0
    input_count = 0
    for role, message in st.session_state.message_history:
		# tiktoken 으로 토큰 수를 계산한다
        token_count = get_message_counts(message)
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count

    # 초기상태에서 System Message에만 이력에 있다면 아직 API 호출이 이루어지지 않은것
    if len(st.session_state.message_history) == 1:
        return

    input_cost = MODEL_PRICES['input'][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICES['output'][st.session_state.model_name] * output_count
    if "gemini" in st.session_state.model_name and (input_count + output_count) > 128000:
        input_cost *= 2
        output_cost *= 2

    cost = output_cost + input_cost

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output cost: ${output_cost:.5f}")


def main():
    init_page()
    init_messages()
    chain = init_chain()

	# 채팅 이력 표시 (2장과 비교해서 조금 위치가 바뀐 것에 주의)
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # 사용자의 입력 모니터링
    if user_input := st.chat_input("궁금한 것을 입력해주세요"):
        st.chat_message('user').markdown(user_input)

		# LLM의 응답을 Streaming 출력
        with st.chat_message('ai'):
            response = st.write_stream(chain.stream({"user_input": user_input}))

		# # 채팅 이력에 추가
        st.session_state.message_history.append(("user", user_input))
        st.session_state.message_history.append(("ai", response))

	# 비용을 계산해서 출력
    calc_and_display_costs()


if __name__ == '__main__':
    main()

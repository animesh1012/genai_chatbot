from nemoguardrails import LLMRails, RailsConfig

def validate_query(input_query,llm):
    #Loading Rails Config
    config = RailsConfig.from_path("./")
    # Configuration of LLMs is passed
    app = LLMRails(config=config,llm=llm)

    # qa_chain = get_qa_chain(app.llm,vector_db)
    # app.register_action(qa_chain, name="qa_chain")
    history = [{"role": "user", "content": input_query}]
    guardrail_msg = app.generate(messages=history)
    return guardrail_msg['content']



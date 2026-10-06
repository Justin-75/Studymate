# src/graphs: LangGraph version of StudyMate
#   schemas.py       what the LLM must return (Pydantic)
#   state.py         what flows between nodes
#   prompts.py       every prompt in one place
#   nodes.py         study-graph nodes (router, retrieve, answer, quiz, grade, ...)
#   study_graph.py   wiring + checkpointer + a terminal demo
#   ingest_graph.py  per-section summaries with a check loop, run once per PDF

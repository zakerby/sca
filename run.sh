#!/bin/bash
# See https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
# Use pure Python parsing (impact performance)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

streamlit run app.py
# src/pipelines.py

def pipeline_a():
    return {
        "name": "pipeline_a",
        "steps": [
            {"name": "extract", "type": "extract", "complexity": "low"},
            {"name": "clean", "type": "transform", "complexity": "medium"},
            {"name": "aggregate", "type": "aggregate", "complexity": "high"},
            {"name": "load", "type": "load", "complexity": "low"},
        ],
    }

def pipeline_b():
    return {
        "name": "pipeline_b",
        "steps": [
            {"name": "extract", "type": "extract", "complexity": "low"},
            {"name": "join", "type": "join", "complexity": "medium"},
            {"name": "transform", "type": "transform", "complexity": "medium"},
            {"name": "aggregate", "type": "aggregate", "complexity": "high"},
            {"name": "load", "type": "load", "complexity": "low"},
        ],
    }

def pipeline_c():
    return {
        "name": "pipeline_c",
        "steps": [
            {"name": "extract", "type": "extract", "complexity": "low"},
            {"name": "transform1", "type": "transform", "complexity": "medium"},
            {"name": "transform2", "type": "transform", "complexity": "medium"},
            {"name": "aggregate", "type": "aggregate", "complexity": "high"},
            {"name": "load", "type": "load", "complexity": "low"},
        ],
    }

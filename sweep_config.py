sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'validation_f1',
        'goal': 'maximize',
    },
    'parameters': {
        'hidden_size': {
            'values':
                [64, 128, 256]
        },
        'dropout': {
            'values':
                [0.0, 0.25, 0.5]
        },
        'model_nr': {
            'value':
                4,
        },
        'data_path': {
            'value':
                "cleaneval.csv",
        },
        'split': {
            'value':
                "55-5-676",
        },
        'num_layers': {
            'value':
                2,
        },
        'learning_rate': {
            'values':
                [0.001, 0.0001, 0.0005]
        },
        'num_epochs': {
            'value':
                100,
        },
        'input_subset': {
            'values':
                ["html_only", "graph_only", "text_only", "html+graph", "html+text", "graph+text", "all"]
            # html_only, graph_only, text_only, html+graph, html+text, graph+text, all
        },
        'scaler': {
            'values':
                ["MinMax", "Off", "Standard"]
            #  ["MinMax", "Off", "Standard"]
        }
    }
}

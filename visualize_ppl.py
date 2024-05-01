#Yuliia Frund

from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def get_ppls(ppl_files: list):
    """
    Specific to our log files: finds the ppl values and the best ppl val and returns them in a nested list.
    :param ppl_files: a list of directories for files that contain the ppl values
    :return: list of lists of vals - a list per file; the last value in every list is the best perplexity value
    """
    ppl_vals = []
    for file in ppl_files:
        with open(file, 'r', encoding='utf-8') as f:
            temp = []
            lines = f.readlines()
            for line in lines:

                if 'ppl:' in line:
                    index_start = line.index('ppl:')
                    index_start = index_start+4
                    index_end = index_start+7
                    val = float(line[index_start:index_end]) #converting these values to float / int makes sure there is no strange behaviour when matplotlib creates figures
                    temp.append(val)

                if 'Best validation result' in line:

                    index_start = line.index('Best validation result')
                    index_start = index_start+51
                    index_end = index_start+5
                    val = float(line[index_start:index_end])
                    temp.append(val)
            ppl_vals.append(temp)

    return ppl_vals

def table(model_names: list, ppl_vals: list):
    """
    Creates a table with perplexity values for every model. Number of models must equal number of top-level items in ppl_vals!
    Saves the table in the same directory.
    :param model_names: names of models to be used as column names
    :param ppl_vals: list of lists (a list per model)
    """
    assert len(model_names) == len(ppl_vals), 'Number of models != number of top-level items in ppl_vals!'

    data = {
        'Validation ppl': [str(i) for i in range(500, 40600, 500)]
    }
    data['Validation ppl'].append('Best')
    for model, item in zip(model_names, ppl_vals):
        data[model.capitalize()] = item


    df = pd.DataFrame(data)
    plt.figure()
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')

    plt.axis('off')

    plt.savefig('validation_perplexity_table.png', bbox_inches='tight', pad_inches=0.05, dpi=500)
    plt.clf()


def line_chart(model_names: list, ppl_vals: list):
    """
    Creates a line chart based on perplexity values for every model. Number of model names must equal numer of top-level items in ppl_vals!
    Saves the line chart in the same directory
    :param model_names: names of models
    :param ppl_vals: list of lists (a list per model)
    """
    assert len(model_names) == len(ppl_vals), 'Number of models != number of top-level items in ppl_vals!'

    for model, item in zip(model_names, ppl_vals):
        x_vals = range(500, 40600, 500)
        plt.plot(x_vals, item[:-1], linestyle='-', label=model.capitalize())
    plt.title('Validation perplexity')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)

    plt.savefig('validation_ppl_linechart.png', dpi=500)
    plt.clf()

def main():
    ppl_vals = get_ppls(['baseline.log','logs/deen_transformer_pre/err', 'logs/deen_transformer_post/err'])

    table(['baseline', 'pre-norm', 'post-norm'], ppl_vals)

    line_chart(['baseline', 'pre-norm', 'post-norm'], ppl_vals)

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython import display


def get_labels(input):
    with h5py.File(f'processed/{input.dset_name}_test.h5', 'r') as f:
        return np.array(f['labels'][()])

def get_indices(input):
    with h5py.File(f'processed/{input.dset_name}_test.h5', 'r') as f:
        return np.array(f['indices'][()])

def to_confusion_matrix(truth, predictions, classes, name):
    t = np.select([truth==i for i in np.unique(truth)],classes, truth)
    p = np.select([predictions==i for i in np.unique(predictions)],classes, predictions)


    cm = confusion_matrix(t, p, labels=np.array(classes))
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

    disp.figure_.savefig(f'{name}.png')

# get accuracy at patch level
def patch_truth(labels, predictions, classes):
    l = labels[:, 0]
    print(classification_report(l, predictions, target_names=classes, digits=4))


# get accuracy at image level
def image_truth(labels, predictions, classes, name):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions], index=['truth', 'image_num', 'predictions']).T
    df = df.groupby(['truth','image_num'])['predictions'].agg(pd.Series.mode).reset_index()
    df = df[pd.notna(pd.to_numeric(df['predictions'], errors='coerce'))]
    df = df.reset_index().drop('image_num', axis=1)

    image_truth = df['truth'].to_numpy().astype(np.uint8)
    image_predictions = df['predictions'].to_numpy().astype(np.uint8)

    cr = classification_report(image_truth, image_predictions, target_names=classes, digits=4, output_dict=True)


    # classification_report_to_csv(cr, name)

    to_confusion_matrix(image_truth, image_predictions, classes, name)


def tuple_gen(labels, predictions, indices):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions, indices], index=['truth', 'image_num', 'predictions', 'indices']).T
    df = df.groupby(['truth', 'image_num']).agg({'predictions' : (lambda x: list(x)), 'indices': (lambda x: list(x))}).reset_index()


    records = df.to_records(index=False)
    results = list(records)

    return results

def viewer(results, classes, index):
    
    size = np.max(results[index][3], axis=0)

    data = [[0 for j in range(int(size[1])+1)] for i in range(int(size[0])+1)]
    for i, idx in enumerate(results[index][3]):
        x, y = int(idx[0]), int(idx[1])
        data[x][y] = results[index][2][i]

    max_data = np.argmax(data, axis=2)

    fig, ax = plt.subplots()
    im = ax.imshow(max_data, cmap='gist_rainbow', vmin=0, vmax=len(classes)-1)

    # Define function to display list when hovering over element
    def on_hover(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            text = ' '.join(str(list(data[y][x])))
            annot.set_text(text)
            annot.xy = (x, y)
            annot.set_visible(True)
            fig.canvas.draw_idle()

    # Add annotation
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)

    # Connect hover event to plot
    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    values = [i for i in range(len(classes))]

    colors = [ im.cmap(im.norm(value)) for value in values]

    patches = [
        mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(classes))
    ]

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    # show plot
    plt.show()


def classification_report_to_barchart(report, name):
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'{name}_report.csv')


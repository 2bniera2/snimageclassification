import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't show all the tensorflow startup messages
import numpy as np
from keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sys import path
import h5py
from data_generator import data_generator
import matplotlib.patches as mpatches



def get_labels(input):
    with h5py.File(f'processed/{input.dset_name}_test.h5', 'r') as f:
        return np.array(f['labels'][()])

def get_indices(input):
    with h5py.File(f'processed/{input.dset_name}_test.h5', 'r') as f:
        return np.array(f['indices'][()])



# get predictions and convert numerical values to class name
def get_predictions(input, classes, model):
    gen = data_generator(
        f'{path[0]}/processed/{input.dset_name}_test.h5',
        'examples',
        classes,
        shuffle=False
    )

    pred = model.predict(gen, use_multiprocessing=True, workers=8)
    return pred, np.argmax(pred, axis=1)

def to_confusion_matrix(truth, predictions, classes):
    t = np.select([truth==i for i in np.unique(truth)],classes, truth)
    p = np.select([predictions==i for i in np.unique(predictions)],classes, predictions)


    cm = confusion_matrix(t, p, labels=np.array(classes))
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()

# get accuracy at patch level
def patch_truth(labels, predictions, classes):
    l = labels[:, 0]
    print(classification_report(l, predictions, target_names=classes, digits=4))

    to_confusion_matrix(l, predictions, classes)

# get accuracy at image level
def image_truth(labels, predictions, classes):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions], index=['truth', 'image_num', 'predictions']).T
    df = df.groupby(['truth','image_num'])['predictions'].agg(pd.Series.mode).reset_index()
    df = df[pd.notna(pd.to_numeric(df['predictions'], errors='coerce'))]
    df = df.reset_index().drop('image_num', axis=1)

    image_truth = df['truth'].to_numpy().astype(np.uint8)
    image_predictions = df['predictions'].to_numpy().astype(np.uint8)

    print(classification_report(image_truth, image_predictions, target_names=classes, digits=4))

    to_confusion_matrix(image_truth, image_predictions, classes)



def prob_truth(labels, predictions, classes):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions], index=['truth', 'image_num', 'predictions']).T
    counts = df.groupby(['truth', 'image_num']).size().reset_index(name='count')
    df = pd.merge(df, counts, on=['truth', 'image_num'])
    df['predictions'] = df['predictions'] / df['count']
    df = df.drop('count', axis=1)
    df = df.groupby(['truth','image_num'])['predictions'].apply(lambda x: np.sum(x.to_numpy(), axis=0)).reset_index(name='predictions')
    df['predictions'] = df['predictions'].apply(lambda x: np.argmax(x))

    image_truth = df['truth'].to_numpy().astype(np.uint8)
    image_predictions = df['predictions'].to_numpy().astype(np.uint8)

    print(classification_report(image_truth, image_predictions, target_names=classes, digits=4))


def tuple_gen(labels, predictions, indices, classes):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions, indices], index=['truth', 'image_num', 'predictions', 'indices']).T
    df = df.groupby(['truth', 'image_num']).agg({'predictions' : (lambda x: list(x)), 'indices': (lambda x: list(x))}).reset_index()


    records = df.to_records(index=False)
    results = list(records)

    return results

def viewer(labels, predictions, indices, classes, index):
    df = pd.DataFrame([labels[:, 0], labels[:, 1], predictions, indices], index=['truth', 'image_num', 'predictions', 'indices']).T
    df = df.groupby(['truth', 'image_num']).agg({'predictions' : (lambda x: list(x)), 'indices': (lambda x: list(x))}).reset_index()


    records = df.to_records(index=False)
    results = list(records)


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





    


def main(model_name, input, classes):
    model = models.load_model(model_name)

    # predictions represented as integer representation of classes
    probs, predictions = get_predictions(input, classes, model)


    # labels with class and image number
    labels = get_labels(input)

    indices = get_indices(input)
    
    # print("==Patch Level==")
    # patch_truth(labels, predictions, classes)

    # print("==Image Level==")
    # image_truth(labels, predictions, classes)

    # print("==Weighted Sum==")
    # prob_truth(labels, probs, classes)

    viewer(labels, probs, indices, classes, 64)





if __name__ == "__main__":
    main()
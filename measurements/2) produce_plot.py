from os import makedirs
from os.path import normpath, join

import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd


images_folder = './images'
plot_name = 'plot.png'
makedirs(images_folder, exist_ok=True)
save_path = normpath(join(images_folder, plot_name))

df_path = normpath("./data/merged_benchmark.csv")
df = pd.read_csv(df_path)
df['size_for_plot'] = [0] * len(df)
df.iloc[df.iloc[:,1] == '112 x 112', -1] = 112 * 112
df.iloc[df.iloc[:,1] == '120 x 160', -1] = 160 * 120
df.iloc[df.iloc[:,1] == '224 x 224', -1] = 224 * 224
df.iloc[df.iloc[:,1] == '256 x 256', -1] = 256 * 256

df.iloc[:, 3] = df.iloc[:, 3] * 1000

#markers = {"IMX500": "s", "Hailo 8l": "c"}
fig = sea.scatterplot(df.iloc[df.iloc[:,-3] == 'picamera2', :],
                      x="Inference time [ms]",
                      y="Inference Energy [mA]",
                      hue="Network",
                      size='size_for_plot',
                      style='device',
#                     markers=markers,
                      )

# Modify Legend to make it more beautiful

fig.legend_.texts[8].set_text("Resolution")
fig.legend_.texts[9].set_text("112 x 112")
fig.legend_.texts[10].set_text("120 x 160")
fig.legend_.texts[11].set_text("224 x 224")
fig.legend_.texts[12].set_text("256 x 256")
fig.legend_.texts[13].set_text("Accelerator")
fig.legend_.texts[15].set_text("Hailo 8L")

sea.move_legend(fig, "lower right", ) #bbox_to_anchor=(1, 1))
sea.set_context("talk")
plt.rcParams['font.size'] = 50


plt.show()
pass


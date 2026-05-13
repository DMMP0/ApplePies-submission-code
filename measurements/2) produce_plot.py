from os import makedirs
from os.path import normpath, join
import plotly.express as px
import pandas as pd


images_folder = './images'
plot_name = 'plot.png'
makedirs(images_folder, exist_ok=True)
save_path = normpath(join(images_folder, plot_name))

df_path = normpath("./data/merged_benchmark.csv")
df = pd.read_csv(df_path)
df['size_for_plot'] = [0] * len(df)
df.iloc[df.iloc[:,1] == '112 x 112', -1] = 112 * 112
df.iloc[df.iloc[:,1] == '120 x 160', -1] = 160 * 160
df.iloc[df.iloc[:,1] == '224 x 224', -1] = 224 * 224
df.iloc[df.iloc[:,1] == '256 x 256', -1] = 256 * 256

df.iloc[:, 3] = df.iloc[:, 3] * 1000


fig = px.scatter(df.iloc[df.iloc[:,-3] == 'picamera2', :], x="Inference time [ms]", y="Inference Energy [mA]", color="Network", size='size_for_plot', symbol="device", )
fig.update_layout(
font=dict(
        size=18,  # Set the font size here
    )
)
fig.show()
pass


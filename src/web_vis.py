
import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from multiprocessing import Queue
from dash.exceptions import PreventUpdate
import time

app = dash.Dash(__name__)
data_queue = Queue()
capture_queue = Queue()

def create_mesh_object(vertices, faces):
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1,
        color='grey'
    )

def create_point_cloud_object(points, colors):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=1
        )
    )

def create_colorscale_plot(data=None):
    if data is None:
        data = [[i for i in range(100)]]

    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=[[0, 'red'], [1, 'white']],
        showscale=False
    ))

    fig.update_layout(
        height=50,
        width=1000,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(showticklabels=False, visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def build_layout():
    camera_dict = {
        'up': {'x': 1, 'y': -1, 'z': 2},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': -0.75, 'y': 0, 'z': -1.75}
    }

    return html.Div([
        html.H1("Defect Visualization", style={'textAlign': 'center'}),
        html.Div([
            dcc.Store(id='update-timestamp', data=0),
            html.Div([
                dcc.Graph(
                    id='3d-visualization-plot',
                    style={'height': '80vh', 'width': '100%'},
                    figure={
                        'data': [],
                        'layout': go.Layout(
                            title='3D Model with Defect Visualization',
                            title_x=0.5,
                            scene={
                                'xaxis': {'visible': False},
                                'yaxis': {'visible': False},
                                'zaxis': {'visible': False},
                                'camera': camera_dict
                            },
                            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                            autosize=True
                        )
                    }
                ),
                dcc.Interval(
                    id='interval-component',
                    interval=1*1000,
                    n_intervals=0
                )
            ], style={
                'width': '75%', 'display': 'flex', 'flexDirection': 'column'
            }),
            html.Div([
                html.H4("Heatmap Color Information", style={'text-align': 'center'}),
                html.Img(
                    id='overlay-image',
                    src='/assets/overlay.png',
                    style={
                        'width': '100%',
                        'max-height': '500px',
                        'object-fit': 'contain',
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                        'border-radius': '5px'
                    }
                ),
                html.Button("Capture New Data", id='capture-button', n_clicks=0,
                            style={
                                'width': '100%', 'background-color': '#007BFF', 'color': 'white',
                                'padding': '12px', 'border-radius': '5px', 'margin-bottom': '10px',
                                'cursor': 'pointer'
                            }),
                dcc.Checklist(
                    options=[{'label': 'Show Defects', 'value': 'SHOW_DEFECTS'}],
                    value=['SHOW_DEFECTS'],
                    id='show-defects-checkbox',
                    style={'margin-bottom': '20px'},
                    labelStyle={'display': 'block', 'margin': '5px', 'color': 'black'}
                ),
            ], style={
                'width': '25%', 'display': 'flex', 'verticalAlign': 'top', 'flexDirection': 'column',
                'padding': '10px',
                'overflowY': 'auto'
            })
        ], style={'display': 'flex','flex': '1', 'overflow': 'hidden' , 'height': '100vh'}),
    ],  style={
            'height': '100vh',
            'display': 'flex',
            'flexDirection': 'column',
            'margin': '0',
            'padding': '0'})

@app.callback(
    Output('3d-visualization-plot', 'figure'),
    [Input('show-defects-checkbox', 'value'),
     Input('interval-component', 'n_intervals')],
    [State('3d-visualization-plot', 'figure')]
)
def update_figure(show_defects, n_intervals, current_figure):
    global data_queue

    if data_queue.empty():
        raise PreventUpdate

    latest_data = data_queue.get()
    mesh_object = \
        create_mesh_object(latest_data['vertices'], latest_data['faces'])

    data = [mesh_object]
    if 'SHOW_DEFECTS' in show_defects:
        for pcd_data in latest_data['pcds']:
            point_cloud_object = \
                create_point_cloud_object(pcd_data['points'], \
                    pcd_data['colors'])
            data.append(point_cloud_object)

    current_figure['data'] = data

    return current_figure

@app.callback(
    Output('capture-button', 'n_clicks'),
    Input('capture-button', 'n_clicks')
)
def capture_new_data(n_clicks):
    if n_clicks > 0:
        capture_queue.put(True)
    return 0  

@app.callback(
    Output('update-timestamp', 'data'),
    [Input('interval-component', 'n_intervals')],
    [State('update-timestamp', 'data')]
)
def refresh_image(n_intervals, current_data):
    if not capture_queue.empty():
        new_timestamp = time.time()
        while not capture_queue.empty():
            capture_queue.get()
        return new_timestamp
    return current_data

@app.callback(
    Output('overlay-image', 'src'),
    [Input('update-timestamp', 'data')]
)
def update_image_src(timestamp):
    if timestamp:
        new_src = f'/assets/overlay.png?t={int(timestamp)}'
        return new_src
    raise PreventUpdate

app.layout = build_layout()

def update_dash_data(intersection_pcds, target_mesh):
    pcd_data = []
    for pcd in intersection_pcds:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd_data.append({'points': points, 'colors': colors})
    
    vertices = np.asarray(target_mesh.vertices)
    faces = np.asarray(target_mesh.triangles)

    data_queue.put({
        'pcds': pcd_data,
        'vertices': vertices,
        'faces': faces
    })
        
def run_dash_app(data_q, capture_q):
    global data_queue, capture_queue
    data_queue = data_q
    capture_queue = capture_q
    app.run_server(debug=False, use_reloader=False, host='0.0.0.0', port=8050)


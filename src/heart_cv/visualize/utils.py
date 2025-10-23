import cv2
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, Button, HBox, VBox, Output, Layout
from IPython.display import display

def make_slice_navigator(draw_slice, num_slices: int):
    """
    Create an IntSlider with + / - buttons to navigate slices.
    `draw_slice(i)` should display the image for index i.
    """
    out = Output()
    slider = IntSlider(min=0, max=num_slices - 1, step=1, value=0,
                       description="z-index", continuous_update=False)

    btn_prev = Button(icon='arrow-left', layout=Layout(min_width='0px',width='auto'))
    btn_next = Button(icon='arrow-right', layout=Layout(min_width='0px',width='auto'))

    def on_prev(_):
        if slider.value > slider.min:
            slider.value -= 1

    def on_next(_):
        if slider.value < slider.max:
            slider.value += 1

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)

    def on_change(change):
        with out:
            out.clear_output(wait=True)
            draw_slice(change["new"])

    slider.observe(on_change, names="value")

    # draw first slice
    with out:
        draw_slice(slider.value)

    display(VBox([HBox([slider, btn_prev, btn_next]), out]))
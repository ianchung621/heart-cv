from typing import Callable

from ipywidgets import IntSlider, Button, HBox, VBox, Output, Layout

def make_z_slider(num_slices: int):
    """
    Create an IntSlider that shows both z-index and portion through the volume.

    Example label: "123 (45.6%)"
    """
    slider: IntSlider = IntSlider(
        min=0, 
        max=num_slices - 1, 
        step=1, 
        value=0,
        description=f"0 (0.0%)",
        continuous_update=False,
        readout=False  # we’ll override description manually
    )

    def update_label(change):
        z = change["new"]
        portion = 100 * z / (num_slices - 1)
        slider.description = f"{z:d} ({portion:5.1f}%)"

    slider.observe(update_label, names="value")
    update_label({"new": slider.value})  # initialize label
    return slider

# --- base layout factory ---
def make_slice_navigator_base(num_slices: int):
    """Return independent UI elements (slider, buttons, outputs)."""

    slider = make_z_slider(num_slices)

    btn_prev: Button = Button(icon='arrow-left', layout=Layout(min_width='0px',width='auto'))
    btn_next: Button = Button(icon='arrow-right', layout=Layout(min_width='0px',width='auto'))

    out_img: Output = Output()
    out_text: Output = Output(layout=Layout(height="120px", background_color="white", padding="6px"))
    out_chart: Output = Output()
    return slider, btn_prev, btn_next, out_img, out_text, out_chart

# --- logic binder ---
def bind_slice_callbacks(
    slider: IntSlider,
    btn_prev: Button,
    btn_next: Button,
    out_img: Output,
    out_text: Output,
    out_chart: Output,
    draw_slice: Callable[[int, Output, Output, Output], None],
):
    """
    Bind interactive logic.
    draw_slice(z_idx, out_img, out_text) should update both outputs.
    """

    def on_prev(_):
        if slider.value > slider.min:
            slider.value -= 1

    def on_next(_):
        if slider.value < slider.max:
            slider.value += 1

    def on_change(change):
        z_idx = change["new"]
        draw_slice(z_idx, out_img, out_text, out_chart)

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    slider.observe(on_change, names="value")

    # draw first slice
    draw_slice(slider.value, out_img, out_text, out_chart)

def make_slice_navigator(draw_slice: Callable[[int, Output, Output, Output], None], num_slices: int):
    """
    Create interactive navigator for slice visualization.
    draw_slice(z_idx, out_img, out_text) will be invoked on change.
    Returns a VBox widget (not displayed automatically).
    """
    slider, btn_prev, btn_next, out_img, out_text, out_chart = make_slice_navigator_base(num_slices)
    bind_slice_callbacks(slider, btn_prev, btn_next, out_img, out_text, out_chart, draw_slice)
    control = HBox([slider, btn_prev, btn_next])
    main_outputs = HBox([out_img, VBox([out_chart, out_text])])
    return VBox([control, main_outputs])
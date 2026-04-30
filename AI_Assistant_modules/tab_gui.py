import os

import gradio as gr

from AI_Assistant_modules.actions.anime_shadow import AnimeShadow
from AI_Assistant_modules.actions.color_scheme import ColorScheme
from AI_Assistant_modules.actions.coloring import Coloring
from AI_Assistant_modules.actions.i2i import Img2Img
from AI_Assistant_modules.actions.lighting import Lighting
from AI_Assistant_modules.actions.line_drawing import LineDrawing
from AI_Assistant_modules.actions.line_drawing_cutout import LineDrawingCutOut
from AI_Assistant_modules.actions.normal_map import NormalMap
from AI_Assistant_modules.actions.resize import ImageResize
from AI_Assistant_modules.actions.stick2body import Stick2Body

# class base_gui:
#  def layout(self, lang_util, transfer_target=None):

def _set_transfer_button(main_tab, target_tab_item, from_tab, transfer_target_tab):
    from_tab.output.transfer_button.click(fn=lambda x: [x, gr.Tabs.update(selected=target_tab_item.id)],
                                          inputs=[from_tab.output.output_image],
                                          outputs=[transfer_target_tab.input_image, main_tab])

def _open_outputdir(app_config):
    dir = os.path.join(app_config.dpath, "output")
    os.makedirs(dir, exist_ok=True)
    # image list
    image_list = []
    for file in os.listdir(dir):
        if file.endswith(".png"):
            image_list.append(file)
    image_list.sort(reverse=True)
    for i, file in enumerate(image_list):
        image_list[i] = (os.path.join(dir, file), file)
    return image_list


# Global Cmd-V/Ctrl-V paste handler. Listens at the document level for
# paste events containing image data, then injects the image into the
# currently visible <input type=file> in the active Gradio tab. Restores
# the "paste a screenshot" workflow that worked on the Windows app.
_PASTE_HANDLER_JS = """
<script>
(function () {
    if (window.__aiAssistantPasteHandler) return;
    window.__aiAssistantPasteHandler = true;

    document.addEventListener('paste', function (event) {
        if (!event.clipboardData) return;
        const items = event.clipboardData.items || [];
        let imageItem = null;
        for (let i = 0; i < items.length; i++) {
            if (items[i].kind === 'file' && items[i].type.startsWith('image/')) {
                imageItem = items[i];
                break;
            }
        }
        if (!imageItem) return;

        const file = imageItem.getAsFile();
        if (!file) return;

        // Find a visible file input that belongs to the active Gradio tab.
        // Gradio renders gr.Image upload areas with a hidden <input type=file>.
        // Pick the first input that lives inside an offsetParent (i.e. visible
        // tab). Skip the mask image if its container also has a normal input.
        const inputs = Array.from(document.querySelectorAll('input[type=file]'))
            .filter(function (el) { return el.offsetParent !== null; });

        // If the user has focused a specific upload area (mouseover its
        // container counts as "active" for our purposes), prefer the closest
        // input. Otherwise use the first visible one.
        let target = null;
        const hovered = document.querySelector('.gradio-container *:hover');
        if (hovered) {
            const container = hovered.closest('[data-testid*=\"image\"], .image-container');
            if (container) {
                target = container.querySelector('input[type=file]');
            }
        }
        if (!target) target = inputs[0];
        if (!target) {
            console.warn('[ai-assistant paste] no visible file input found');
            return;
        }

        const dt = new DataTransfer();
        dt.items.add(file);
        target.files = dt.files;
        target.dispatchEvent(new Event('change', { bubbles: true }));
        console.log('[ai-assistant paste] injected', file.name || 'image', 'into', target);
        event.preventDefault();
    });
})();
</script>
"""


def gradio_tab_gui(app_config):
    lang_util = app_config.lang_util

    with gr.Blocks(title="AI_Assistant") as main_block:
        # Inject the global paste handler. gr.HTML renders inline; the
        # <script> runs once on page load.
        gr.HTML(_PASTE_HANDLER_JS)
        with gr.Tabs() as main_tab:
            with gr.TabItem(lang_util.get_text("img2img")):
                img_2_img = Img2Img(app_config)
                img_2_img.layout("transfer_to_lineart")
            with gr.TabItem(lang_util.get_text("stick2body")):
                stick_2_body = Stick2Body(app_config)
                stick_2_body.layout()        
            with gr.TabItem(lang_util.get_text("lineart"), id="lineart") as line_drawing_tab_item:
                line_drawing_tab = LineDrawing(app_config)
                line_drawing_tab.layout("transfer_to_normalmap")
            with gr.TabItem(lang_util.get_text("lineart2")):
                line_drawing_cutout_tab = LineDrawingCutOut(app_config)
                line_drawing_cutout_tab.layout("transfer_to_normalmap")
            with gr.TabItem(lang_util.get_text("normalmap"), id="normalmap") as normal_map_tab_item:
                normal_map = NormalMap(app_config)
                normal_map.layout("transfer_to_lighting")
            with gr.TabItem(lang_util.get_text("lighting"), id="lighting") as lighting_tab_item:
                lighting = Lighting(app_config)
                lighting.layout("anime_shadow_tab_transfer")
            with gr.TabItem(lang_util.get_text("anime_shadow"), id="anime_shadow") as anime_shadow_tab_item:
                anime_shadow = AnimeShadow(app_config)
                anime_shadow.layout()
            with gr.TabItem(lang_util.get_text("color_scheme")):
                color_scheme = ColorScheme(app_config)
                color_scheme.layout()
            with gr.TabItem(lang_util.get_text("coloring")):
                coloring = Coloring(app_config)
                coloring.layout()        
            with gr.TabItem(lang_util.get_text("resize")):
                ImageResize(app_config).layout()
            if app_config.device == "cloud" or app_config.device == "docker":
                with gr.TabItem(lang_util.get_text("output_destination")) as output_tab_item:
                    gallery = gr.Gallery([], label=lang_util.get_text("output_destination"), interactive=False,
                                         height="85vh")
                    output_tab_item.select(fn=lambda: _open_outputdir(app_config), outputs=[gallery])

        # タブ間転送の動作設定
        _set_transfer_button(main_tab, line_drawing_tab_item, img_2_img, line_drawing_tab)
        _set_transfer_button(main_tab, normal_map_tab_item, line_drawing_tab, normal_map)
        _set_transfer_button(main_tab, normal_map_tab_item, line_drawing_cutout_tab, normal_map)
        _set_transfer_button(main_tab, lighting_tab_item, normal_map, lighting)
        _set_transfer_button(main_tab, anime_shadow_tab_item, lighting, anime_shadow)
 
    return main_block

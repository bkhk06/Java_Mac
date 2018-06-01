import moviepy.editor as mpy
import skimage.exposure as ske
import skimage.filters as skf

clip = mpy.VideoFileClip("sinc.gif")
gray = clip.fx(mpy.vfx.blackwhite).to_mask()


def apply_effect(effect, label, **kw):
    """ Returns a clip with the effect applied and a top label"""
    filtr = lambda im: effect(im, **kw)
    new_clip = gray.fl_image(filtr).to_RGB()
    txt = (mpy.TextClip(label, font="Amiri-Bold", fontsize=25,
                        bg_color='white', size=new_clip.size)
           .set_position(("center"))
           .set_duration(1))
    return mpy.concatenate_videoclips([txt, new_clip])


equalized = apply_effect(ske.equalize_hist, "Equalized")
rescaled = apply_effect(ske.rescale_intensity, "Rescaled")
adjusted = apply_effect(ske.adjust_log, "Adjusted")
blurred = apply_effect(skf.gaussian_filter, "Blurred", sigma=4)

clips = [equalized, adjusted, blurred, rescaled]
animation = mpy.concatenate_videoclips(clips)
animation.write_gif("sinc_cat.gif", fps=15)
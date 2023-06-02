from manim import *


class DragAndDrop(Scene):
    def construct(self):
        # Create a simple rectangular phone
        phone = RoundedRectangle(
            corner_radius=0.2, height=5, width=3, fill_color=WHITE, fill_opacity=1)

        # Create an attached side menu
        menu = RoundedRectangle(
            corner_radius=0.1, height=4, width=1, fill_color=LIGHT_GRAY, fill_opacity=1)
        menu.next_to(phone, LEFT, buff=0)

        # Group the phone and menu together
        phone_and_menu = VGroup(phone, menu)
        self.play(Create(phone_and_menu))

        # Create an input box labeled "Text" in the menu
        input_box = RoundedRectangle(
            corner_radius=0.1, height=0.5, width=0.8, fill_color=BLUE, fill_opacity=1)
        input_box_text = Text("Text").scale(0.2).next_to(
            input_box, DOWN, buff=0.1).set_color(BLACK)
        input_box_group = VGroup(input_box, input_box_text)
        input_box_group.next_to(menu, UP, buff=0.2)
        self.play(Create(input_box_group))

        # Animate the dragging of the input box onto the main screen
        self.play(
            input_box_group.animate.next_to(phone, UP, buff=0.2),
            FadeOut(menu)
        )

        # Animate the input box enlarging to the size of the phone's width and moving to the center
        target_width = phone.width - 0.4
        target_scale_factor = target_width / input_box.width
        self.play(
            input_box_group.animate.move_to(phone.get_center()),
        )

        self.wait()

# To render the animation, use the following command:
# manim -p -ql file_name.py DragAndDrop

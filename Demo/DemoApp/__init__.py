from otree.api import *
import threading
import os
import shutil
from .videoTranscriberV2 import LiveMeetingAnalyzer


class Constants(BaseConstants):
    name_in_url = 'demoapp'
    players_per_group = None
    num_rounds = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    video_path = models.StringField()
    video_filename = models.StringField()
    num_speakers = models.IntegerField(min=2, max=8)


def perform_analysis(video_path):
    try:
        analyzer = LiveMeetingAnalyzer(video_path)
        analyzer.analyze_meeting()
    except Exception as e:
        print(f"Analysis error: {str(e)}")


class MyPage(Page):
    form_model = 'player'
    form_fields = ['video_path']

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        if player.video_path:
            try:
                # Get the original filename
                original_filename = os.path.basename(player.video_path)

                # Create static directory path
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                static_dir = os.path.join(base_dir, '_static', 'DemoApp')

                # Create directories if they don't exist
                os.makedirs(static_dir, exist_ok=True)

                # Copy file to static directory
                dest_path = os.path.join(static_dir, original_filename)
                shutil.copy2(player.video_path, dest_path)

                # Store the filename for use in Results page
                player.video_filename = original_filename

                print(f"Successfully copied video to: {dest_path}")

            except Exception as e:
                print(f"Error copying video file: {str(e)}")
                raise e


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        video_path = os.path.join('_static', 'DemoApp', player.video_filename)
        threading.Thread(
            target=perform_analysis,
            args=(video_path,),
            daemon=True
        ).start()

        return {
            "video_path": f"DemoApp/{player.video_filename}"
        }


page_sequence = [MyPage, Results]
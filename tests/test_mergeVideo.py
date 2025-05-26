import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now we can import modules from src
from video import video as VideoModule_Video # Renaming to avoid conflict
from mergeVideo import check_if_chimeric_tracks_needed, augment_merge_file_with_tracks
# Import tools to mock its attributes
from src import tools as tools_module


def create_mock_video_object(file_path="dummy.mkv", audios=None, subtitles=None, chimeric_files=None):
    """Helper function to create a mock video object with specified audio/subtitle tracks."""
    # We need to mock the __init__ of VideoModule_Video or its dependencies if it does file checks
    # For simplicity, we'll create a MagicMock that has the necessary attributes.
    # If we were to instantiate VideoModule_Video, we'd need to handle its file existence checks.
    
    mock_video = MagicMock(spec=VideoModule_Video)
    mock_video.filePath = file_path
    mock_video.fileBaseName = os.path.splitext(os.path.basename(file_path))[0]
    mock_video.audios = audios if audios is not None else {}
    mock_video.subtitles = subtitles if subtitles is not None else {}
    mock_video.chimeric_files = chimeric_files if chimeric_files is not None else [] # Default to empty list
    return mock_video

class TestMergeVideoChimericHelpers(unittest.TestCase):

    # --- Tests for check_if_chimeric_tracks_needed ---

    def test_check_if_chimeric_tracks_needed_no_chimeric_files(self):
        """Test Case 2.1: V_abs_best_obj.chimeric_files is None or empty."""
        out_metadata_obj = create_mock_video_object(audios={'eng': []}, subtitles={'eng': []})
        
        V_abs_best_obj_none = create_mock_video_object(chimeric_files=None)
        self.assertEqual(check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj_none), [])

        V_abs_best_obj_empty = create_mock_video_object(chimeric_files=[])
        self.assertEqual(check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj_empty), [])

    def test_check_if_chimeric_tracks_needed_no_new_tracks(self):
        """Test Case 2.2: Chimeric files have no new tracks."""
        out_metadata_obj = create_mock_video_object(
            audios={'eng': [{'Language': 'eng', '@type': 'Audio', 'StreamOrder': '1'}]},
            subtitles={'eng': [{'Language': 'eng', '@type': 'Text', 'StreamOrder': '2'}]}
        )
        chimeric_vid_obj1 = create_mock_video_object(
            file_path="chimeric1.mkv",
            audios={'eng': [{'Language': 'eng', '@type': 'Audio', 'StreamOrder': '1c'}]},
            subtitles={'eng': [{'Language': 'eng', '@type': 'Text', 'StreamOrder': '2c'}]}
        )
        V_abs_best_obj = create_mock_video_object(chimeric_files=[chimeric_vid_obj1])
        self.assertEqual(check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj), [])

    def test_check_if_chimeric_tracks_needed_new_audio_track(self):
        """Test Case 2.3: Chimeric file has new audio track."""
        out_metadata_obj = create_mock_video_object(audios={'eng': []})
        fra_audio_track = {'StreamOrder': '1c', 'Language': 'fra', '@type': 'Audio', 'Title': 'French Stereo'}
        chimeric_vid_obj1 = create_mock_video_object(file_path="chimeric1.mkv", audios={'fra': [fra_audio_track]})
        V_abs_best_obj = create_mock_video_object(chimeric_files=[chimeric_vid_obj1])
        
        result = check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], chimeric_vid_obj1)
        self.assertEqual(len(result[0][1]), 1)
        self.assertDictEqual(result[0][1][0], fra_audio_track)

    def test_check_if_chimeric_tracks_needed_new_subtitle_track(self):
        """Test Case 2.4: Chimeric file has new subtitle track."""
        out_metadata_obj = create_mock_video_object(subtitles={'eng': []})
        spa_subtitle_track = {'StreamOrder': '2c', 'Language': 'spa', '@type': 'Text', 'Title': 'Spanish Sub'}
        chimeric_vid_obj1 = create_mock_video_object(file_path="chimeric1.mkv", subtitles={'spa': [spa_subtitle_track]})
        V_abs_best_obj = create_mock_video_object(chimeric_files=[chimeric_vid_obj1])

        result = check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], chimeric_vid_obj1)
        self.assertEqual(len(result[0][1]), 1)
        self.assertDictEqual(result[0][1][0], spa_subtitle_track)

    def test_check_if_chimeric_tracks_needed_multiple_files_mixed_tracks(self):
        """Test Case 2.5: Multiple chimeric files, mixed new tracks."""
        out_metadata_obj = create_mock_video_object(
            audios={'eng': [{'Language': 'eng', '@type': 'Audio', 'StreamOrder': '1'}]},
            subtitles={'eng': [{'Language': 'eng', '@type': 'Text', 'StreamOrder': '2'}]}
        )
        
        fra_audio_track = {'StreamOrder': 'a1', 'Language': 'fra', '@type': 'Audio'}
        chimeric_vid_obj1 = create_mock_video_object(file_path="chimeric1.mkv", audios={'fra': [fra_audio_track]})
        
        spa_subtitle_track = {'StreamOrder': 's1', 'Language': 'spa', '@type': 'Text'}
        chimeric_vid_obj2 = create_mock_video_object(file_path="chimeric2.mkv", subtitles={'spa': [spa_subtitle_track]})
        
        deu_audio_track = {'StreamOrder': 'a2', 'Language': 'deu', '@type': 'Audio'}
        ita_subtitle_track = {'StreamOrder': 's2', 'Language': 'ita', '@type': 'Text'}
        chimeric_vid_obj3 = create_mock_video_object(file_path="chimeric3.mkv", 
                                                  audios={'deu': [deu_audio_track]}, 
                                                  subtitles={'ita': [ita_subtitle_track]})
        # Add a chimeric file with a language already added by a previous chimeric file, to test if it's correctly skipped for that language
        chimeric_vid_obj4 = create_mock_video_object(file_path="chimeric4.mkv", audios={'fra': [{'StreamOrder': 'a3', 'Language': 'fra', '@type': 'Audio'}]})


        V_abs_best_obj = create_mock_video_object(chimeric_files=[chimeric_vid_obj1, chimeric_vid_obj2, chimeric_vid_obj3, chimeric_vid_obj4])
        
        result = check_if_chimeric_tracks_needed(out_metadata_obj, V_abs_best_obj)
        self.assertEqual(len(result), 3, "Should be 3 entries: one for chimeric1, one for chimeric2, one for chimeric3. chimeric4's 'fra' audio is now duplicate.")

        found_ch1, found_ch2, found_ch3 = False, False, False
        for res_obj, res_tracks in result:
            if res_obj.filePath == "chimeric1.mkv":
                self.assertEqual(len(res_tracks), 1)
                self.assertIn(fra_audio_track, res_tracks)
                found_ch1 = True
            elif res_obj.filePath == "chimeric2.mkv":
                self.assertEqual(len(res_tracks), 1)
                self.assertIn(spa_subtitle_track, res_tracks)
                found_ch2 = True
            elif res_obj.filePath == "chimeric3.mkv":
                self.assertEqual(len(res_tracks), 2) # deu audio and ita sub
                self.assertIn(deu_audio_track, res_tracks)
                self.assertIn(ita_subtitle_track, res_tracks)
                found_ch3 = True
        
        self.assertTrue(found_ch1 and found_ch2 and found_ch3, "Not all expected chimeric files/tracks were found.")

    # --- Tests for augment_merge_file_with_tracks ---
    @patch('src.mergeVideo.tools_module.launch_cmdExt') # Mock launch_cmdExt from mergeVideo's perspective
    @patch('src.mergeVideo.tools_module.software')    # Mock tools.software from mergeVideo's perspective
    def test_augment_merge_file_with_tracks_command_generation(self, mock_tools_software, mock_launch_cmdExt):
        """Test Case 3.1: Verify mkvmerge command generation."""
        mock_tools_software.__getitem__.side_effect = lambda key: '/usr/bin/mkvmerge' if key == 'mkvmerge' else None
        
        original_temp_path = "/tmp/merged_step1.mkv"
        V_abs_best_basename = "MyBestVideo"
        tools_temp_folder = "/tmp"

        ch1_fra_audio = {'StreamOrder': '1', 'Language': 'fra', '@type': 'Audio', 'Title': 'French Audio'}
        ch1_obj = create_mock_video_object(file_path="/chimeras/ch1.mkv", audios={'fra': [ch1_fra_audio]})

        ch2_spa_sub = {'StreamOrder': '5', 'Language': 'spa', '@type': 'Text', 'Title': 'Spanish Subtitle'}
        ch2_obj = create_mock_video_object(file_path="/chimeras/ch2.mkv", subtitles={'spa': [ch2_spa_sub]})
        
        ch3_deu_audio = {'StreamOrder': '2', 'Language': 'deu', '@type': 'Audio', 'Title': 'German Audio'}
        ch3_ita_sub = {'StreamOrder': '3', 'Language': 'ita', '@type': 'Text', 'Title': 'Italian Subtitle'}
        ch3_obj = create_mock_video_object(file_path="/chimeras/ch3.mkv", 
                                         audios={'deu': [ch3_deu_audio]}, 
                                         subtitles={'ita': [ch3_ita_sub]})

        needed_tracks_info_list = [
            (ch1_obj, [ch1_fra_audio]),
            (ch2_obj, [ch2_spa_sub]),
            (ch3_obj, [ch3_deu_audio, ch3_ita_sub])
        ]

        expected_augmented_path = os.path.join(tools_temp_folder, f"{V_abs_best_basename}_merged_step2.mkv")
        
        returned_path = augment_merge_file_with_tracks(original_temp_path, needed_tracks_info_list, tools_temp_folder, V_abs_best_basename)
        self.assertEqual(returned_path, expected_augmented_path)

        mock_launch_cmdExt.assert_called_once()
        actual_cmd = mock_launch_cmdExt.call_args[0][0]

        # Basic command structure checks
        self.assertEqual(actual_cmd[0], '/usr/bin/mkvmerge')
        self.assertIn('-o', actual_cmd)
        self.assertEqual(actual_cmd[actual_cmd.index('-o') + 1], expected_augmented_path)
        self.assertIn(original_temp_path, actual_cmd)
        
        # Check for ch1.mkv processing
        self.assertIn(ch1_obj.filePath, actual_cmd)
        ch1_idx = actual_cmd.index(ch1_obj.filePath)
        self.assertIn('--audio-tracks', actual_cmd[ch1_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--audio-tracks', ch1_idx) + 1], ch1_fra_audio['StreamOrder'])
        self.assertIn('--language', actual_cmd[ch1_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--language', ch1_idx) + 1], f"{ch1_fra_audio['StreamOrder']}:{ch1_fra_audio['Language']}")
        self.assertIn('--track-name', actual_cmd[ch1_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--track-name', ch1_idx) + 1], f"{ch1_fra_audio['StreamOrder']}:{ch1_fra_audio['Title']}")
        self.assertIn('--no-video', actual_cmd[ch1_idx:])
        self.assertIn('--no-subtitles', actual_cmd[ch1_idx:]) # Since only audio is mapped from ch1

        # Check for ch2.mkv processing
        self.assertIn(ch2_obj.filePath, actual_cmd)
        ch2_idx = actual_cmd.index(ch2_obj.filePath)
        self.assertIn('--subtitle-tracks', actual_cmd[ch2_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--subtitle-tracks', ch2_idx) + 1], ch2_spa_sub['StreamOrder'])
        self.assertIn('--language', actual_cmd[ch2_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--language', ch2_idx) + 1], f"{ch2_spa_sub['StreamOrder']}:{ch2_spa_sub['Language']}")
        self.assertIn('--no-audio', actual_cmd[ch2_idx:]) # Since only subs are mapped from ch2

        # Check for ch3.mkv processing (both audio and subtitle)
        self.assertIn(ch3_obj.filePath, actual_cmd)
        ch3_idx = actual_cmd.index(ch3_obj.filePath)
        self.assertIn('--audio-tracks', actual_cmd[ch3_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--audio-tracks', ch3_idx) + 1], ch3_deu_audio['StreamOrder'])
        self.assertIn('--subtitle-tracks', actual_cmd[ch3_idx:])
        self.assertEqual(actual_cmd[actual_cmd.index('--subtitle-tracks', ch3_idx) + 1], ch3_ita_sub['StreamOrder'])
        
        # Check for one of the language flags for ch3 (deu audio)
        deu_lang_flag_idx = -1
        for i, token in enumerate(actual_cmd[ch3_idx:]):
            if token == '--language' and actual_cmd[ch3_idx+i+1] == f"{ch3_deu_audio['StreamOrder']}:{ch3_deu_audio['Language']}":
                deu_lang_flag_idx = ch3_idx+i
                break
        self.assertNotEqual(deu_lang_flag_idx, -1, "German language flag for ch3 not found correctly.")

        # Check for one of the track name flags for ch3 (ita sub)
        ita_name_flag_idx = -1
        for i, token in enumerate(actual_cmd[ch3_idx:]):
            if token == '--track-name' and actual_cmd[ch3_idx+i+1] == f"{ch3_ita_sub['StreamOrder']}:{ch3_ita_sub['Title']}":
                ita_name_flag_idx = ch3_idx+i
                break
        self.assertNotEqual(ita_name_flag_idx, -1, "Italian subtitle name flag for ch3 not found correctly.")

        # Check for default and forced flags (example for ch1_fra_audio)
        # These checks need to find the flags associated with ch1_fra_audio['StreamOrder'] within the segment for ch1.mkv
        ch1_segment_end = actual_cmd.index(')', ch1_idx) 
        
        self.assertTrue(any(actual_cmd[i] == '--default-track-flag' and actual_cmd[i+1] == f"{ch1_fra_audio['StreamOrder']}:0" 
                            for i in range(ch1_idx, ch1_segment_end -1)))
        self.assertTrue(any(actual_cmd[i] == '--forced-display-flag' and actual_cmd[i+1] == f"{ch1_fra_audio['StreamOrder']}:0"
                            for i in range(ch1_idx, ch1_segment_end -1)))
        
        # Check general flags per input group
        self.assertTrue(all(flag in actual_cmd[ch1_idx:ch1_segment_end] for flag in ['--no-attachments', '--no-global-tags', '--no-chapters']))


if __name__ == '__main__':
    unittest.main()

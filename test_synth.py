"""A suite of unittests for testing an implementation of the CPSLP TTS assignment prior 
to submitting it.  This set of tests will ensure the *API* works in the way expected by the automatic
marking code, so you must run them before finally submitting your code.  Specifically, they
test 1) basic requirements of your code that are needed in order for the automatic marking to work
and 2) whether it appears each Extension task appears to have been implemented and enabled.

Note: These tests only indicate whether the automatic marking tests will be able to evaluate your code.
None of these will test whether your code gives the correct answers or not.  If you like, you could add
extra tests of your own here to automatically test your code as you develop.  But just be sure not to change
the CPSLP ones which check your code can be called as expected.

Final Caution: the automatic marking assumes your code will pass the tests below, so do test your API properly!"""

import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from numpy import array_equal

import simpleaudio
import synth as student_code  # student's own implementation that we are testing in this module


def default_args():
    return Namespace(diphones='./diphones', play=False, outfile='testout.wav', phrase='shimmer',
                     volume=None, spell=False, reverse=None, fromfile=None, crossfade=False)

def synthesise_to_wav(args):

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir) / "testout.wav"
        args.outfile = str(outpath)
        student_code.main(args)
        a = simpleaudio.Audio()
        a.load(str(outpath))

    return a


class TestSynth(unittest.TestCase):

    def test_basic_diphone_dir_handling(self):
        """Does the code respect the location of the diphones directory to use correctly?"""

        args = default_args()
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_wavfile = Path(tmpdir) / "i_i.wav"
            with open(dummy_wavfile, 'w') as outf:
                outf.write("this is a purposely corrupted wav - it should not load")

            args.diphones = tmpdir
            self.assertRaises(Exception, synthesise_to_wav, args)

    def test_basic_utterance_creation(self):
        """Can we create an instance of the Utterance class correctly?"""

        utt = student_code.Utterance('connection')
        self.assertTrue(isinstance(utt.get_phone_seq(), list), 'Utterance.get_phone_seq should return a list')
        self.assertTrue(isinstance(utt.get_phone_seq()[0], str), 'Utterance.get_phone_seq should return a list of strings')


    def test_basic_code_saves_waveforms(self):
        """Does the code save a waveform to the given file name when the outfile flag is given?"""

        args = default_args()
        try:
            synthesise_to_wav(args)
        except FileNotFoundError as err:
            raise FileNotFoundError('No waveform was saved to the requested path') from err
        except EOFError:
            self.fail('Waveform file that was saved was empty')

    def test_basic_no_save_to_diphonedir(self):
        """Does the code add any files to the diphone directory? (It shouldn't!)"""

        args = default_args()
        synthesise_to_wav(args)

        # we expect only 1619 files to be in the diphones directory (otherwise the
        # code has meddled with the directory!)
        self.assertEqual(len(list(Path(args.diphones).iterdir())), 1619, 'Your code should not add/remove files in the diphones directory')

    def test_basic_synthesis_matches_example(self):
        """Does the code synthesis a basic utterance to match the waveform in the examples directory?"""

        args = default_args()
        args.phrase = 'a rose by any other name would smell as sweet'
        w = synthesise_to_wav(args)
        ref = simpleaudio.Audio()
        ref.load('./examples/basic/rose.wav')
        self.assertTrue(array_equal(w.data, ref.data),
                        f'Your code does not create a waveform the same as ./examples/basic/rose.wav for the phrase "{args.phrase}"')

    def test_volume_ext_enabled(self):
        """Does the code appear to have Volume Ext implemented and enabled by default?"""

        args = default_args()
        wav_without_vol = synthesise_to_wav(args)
        args.volume = 20
        wav_with_vol = synthesise_to_wav(args)

        # the maximum value in each case should be different
        self.assertNotEqual(wav_without_vol.data.max(), wav_with_vol.data.max(), 'It appears Ext "Volume" is not implemented/enabled')

    def test_reverse_ext_enabled(self):
        """Does the code appear to have Backwards Ext implemented and enabled by default?"""

        args = default_args()
        args.phrase="that is"
        all_wavs = {'no reverse': synthesise_to_wav(args)}

        for flag in ['words', 'phones', 'signal']:
            args.reverse = flag
            all_wavs[flag] = synthesise_to_wav(args)

        # simple check - total sum of waveform samples in each condition should be different in all cases
        # (no duplicates) except in the case of signal reversal (which should be same samples)
        totals = [sum(w.data) for w in all_wavs.values()]
        self.assertEqual(len(totals)-1, len(set(totals)), 'It appears not all parts of Ext "Reverse" are implemented/enabled')
        print(all_wavs['signal'].data[::-1],all_wavs['no reverse'].data)
        self.assertEqual(sum(all_wavs['signal'].data[::-1]-all_wavs['no reverse'].data), 0)


    def test_filesynth_ext_enabled(self):
        """Does the code appear to have Ext "Filesynth" implemented and enabled by default?"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_txt_file = Path(tmpdir) / "temp.txt"
            with open(str(tmp_txt_file), 'w') as textfile:
                textfile.write("Hello.\nHow are you?")

            args = default_args()
            args.phrase, args.fromfile = None, str(tmp_txt_file)

            try:
                synthesise_to_wav(args)
            except:
                raise NotImplementedError('It appears Ext "Filesynth" is not implemented/enabled')


    def test_dates_ext_enabled(self):
        """Does the code appear to have Ext "Dates" implemented and enabled by default?"""

        args = default_args()
        args.phrase = "today is 09/03/1973"
        try:
            wav_numeric_date = synthesise_to_wav(args)
        except:
            self.fail('It appears Ext "Dates" is not implemented/enabled')

        args.phrase = "today is march ninth nineteen seventy three"
        wav_words_date = synthesise_to_wav(args)

        self.assertTrue(array_equal(wav_numeric_date.data, wav_words_date.data),
                        'It appears Ext "Dates" is not implemented/enabled')
        
    def test_crossfade_ext_enabled(self):
        """Does the code appear to have Ext "Crossfade" implemented and enabled by default?"""

        args = default_args()
        wav_without_flag = synthesise_to_wav(args)
        args.crossfade = True
        wav_with_flag = synthesise_to_wav(args)

        # waveform should have different synthesised waveform in both conditions
        self.assertNotEqual(sum(wav_with_flag.data), sum(wav_without_flag.data),
                            'It appears Ext "Crossfade" is not implemented/enabled')

    

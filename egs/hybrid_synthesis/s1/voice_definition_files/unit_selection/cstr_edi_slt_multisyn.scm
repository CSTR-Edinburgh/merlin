;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                       ;;
;;;                Centre for Speech Technology Research                  ;;
;;;                     University of Edinburgh, UK                       ;;
;;;                       Copyright (c) 2003, 2004                        ;;
;;;                        All Rights Reserved.                           ;;
;;;                                                                       ;;
;;;  Permission is hereby granted, free of charge, to use and distribute  ;;
;;;  this software and its documentation without restriction, including   ;;
;;;  without limitation the rights to use, copy, modify, merge, publish,  ;;
;;;  distribute, sublicense, and/or sell copies of this work, and to      ;;
;;;  permit persons to whom this work is furnished to do so, subject to   ;;
;;;  the following conditions:                                            ;;
;;;   1. The code must retain the above copyright notice, this list of    ;;
;;;      conditions and the following disclaimer.                         ;;
;;;   2. Any modifications must be clearly marked as such.                ;;
;;;   3. Original authors' names are not deleted.                         ;;
;;;   4. The authors' names are not used to endorse or promote products   ;;
;;;      derived from this software without specific prior written        ;;
;;;      permission.                                                      ;;
;;;                                                                       ;;
;;;  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        ;;
;;;  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ;;
;;;  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ;;
;;;  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     ;;
;;;  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ;;
;;;  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ;;
;;;  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ;;
;;;  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ;;
;;;  THIS SOFTWARE.                                                       ;;
;;;                                                                       ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; slt Multisyn Voice definition for use with Unilex
;;;


;; SCHEME PATHS
(defvar multisyn_lib_dir (path-append libdir "multisyn/"))
(defvar slt_edi_multisyn_dir (cdr (assoc 'cstr_edi_slt_multisyn voice-locations)))
(defvar slt_edi_data_dir (path-append slt_edi_multisyn_dir "slt"))


;; These may or may not be already loaded.
(if (not (member_string multisyn_lib_dir libdir))
    (set! load-path (cons multisyn_lib_dir load-path)))
(if (not (member_string 'unilex-rpx (lex.list)))
    (load (path-append lexdir "unilex/" (string-append 'unilex-rpx ".scm"))))

;; REQUIRES
(require 'unilex_phones)
(require 'multisyn)
(require 'phrase)
(require 'pos)

;; DATA PATHS

;; Location of Base utterances
(set! slt_edi_base_dirs (list (path-append slt_edi_data_dir "utt/")
			   (path-append slt_edi_data_dir "wav/")
			   (path-append slt_edi_data_dir "pm/")
			   (path-append slt_edi_data_dir "coef/")
			   ".utt" ".wav" ".pm" ".coef"))

(make_voice_definition 'cstr_edi_slt_multisyn 
		       16000
		       'voice_cstr_edi_slt_multisyn_configure
		       unilex-rpx-backoff_rules
		       slt_edi_data_dir
		       (list (list slt_edi_base_dirs "utts.data")
			     (list slt_edi_base_dirs "slt_pauses.data")))

(define (voice_cstr_edi_slt_multisyn_configure_pre voice)
  "(voice_cstr_edi_slt_multisyn_configure_pre voice)
 Preconfiguration for female British English (slt) for
 the multisyn unitselection engine with Combilex."
  (voice_reset)
  (Parameter.set 'Language 'britishenglish)
  (Param.set 'Ignore_Bad_Phones t) ;; has no effect; but see voice.init in
                                   ;; festival/lib/multisyn/multisyn.scm
  (unilex::select_phoneset))


(define (voice_cstr_edi_slt_multisyn_configure voice)
  "(voice_cstr_edi_slt_multisyn_configure voice)
 Set up the current voice to be female British English (slt) for
 the multisyn unitselection engine with Combilex."
  (let (this_voice)
    (voice_reset)
    (Parameter.set 'Language 'britishenglish)
    (unilex::select_phoneset)
    (set! token_to_words english_token_to_words)
    (set! pos_lex_name "english_poslex")
    (set! pos_ngram_name 'english_pos_ngram)
    (set! pos_supported t)
    (set! guess_pos english_guess_pos)
    (lex.select 'unilex-rpx)
    (set! postlex_rules_hooks (list postlex_apos_s_check 
                                    postlex_the_vs_thee
                                    postlex_intervoc_r
                                    postlex_a))
    (set! postlex_vowel_reduce_table nil)
    (Parameter.set 'Word_Method Unilex_Word)
    (Parameter.set 'Phrase_Method 'cart_tree)
    (set! phrase_cart_tree simple_phrase_cart_tree)
    (Parameter.set 'Pause_Method MultiSyn_Pauses)
    (Parameter.set 'Int_Method nil)
    (Parameter.set 'Int_Target_Method nil)
    (Parameter.set 'Duration_Method nil)
    (Param.set 'Synth_Method 'MultiSyn)
    (Param.set 'unisyn.window_symmetric 0));;)

    (du_voice.set_target_cost_weight voice 0.5)
    ;;(du_voice.set_target_cost_weight voice 0.1) ;;TOM MERRITT
    ;;(du_voice.set_target_cost_weight voice 0.0)
)

(proclaim_voice
 'cstr_edi_slt_multisyn
 '((language english)(gender female)(dialect british)
   (description "slt multisyn unitsel voice  (Combilex configuration).")))

(provide 'cstr_edi_slt_multisyn)

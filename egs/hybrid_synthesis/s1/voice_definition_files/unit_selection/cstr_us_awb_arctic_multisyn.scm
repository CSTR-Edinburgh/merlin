;; Awb Voice definition

;; backoff rules
(set! us_awb-backoff_rules
  '(
    (aa ax)
    (ae ax)
    (ah ax)
    (ao ax)
    (aw ax)
    (axr ax)
    (ay ax)
    (eh ax)
    (er ax)
    (ey ax)
    (ih ax)
    (iy ax)
    (ow ax)
    (oy ax)
    (uh ax)
    (uw ax)
    (_ #)))


;; SCHEME PATHS

(defvar multisyn_lib_dir (path-append libdir "multisyn/"))
(defvar us_awb_multisyn_dir (cdr (assoc 'cstr_us_awb_arctic_multisyn voice-locations)))
(defvar us_awb_data_dir (path-append us_awb_multisyn_dir "awb"))

;; These may or may not be already loaded.
(if (not (member_string multisyn_lib_dir libdir))
    (set! load-path (cons multisyn_lib_dir load-path)))

;; REQUIRES
(require 'radio_phones_multisyn)
(require 'multisyn)
(require 'phrase)
(require 'pos)

;; DATA PATHS

;; Location of Base utterances
(set! us_awb_base_dirs (list (path-append us_awb_data_dir "utt/")
			     (path-append us_awb_data_dir "lpc/")
			     (path-append us_awb_data_dir "lpc/")
			     (path-append us_awb_data_dir "coef/")
			   ".utt" ".res" ".lpc" ".coef"))

(make_voice_definition 'cstr_us_awb_arctic_multisyn 
		       16000
		       'voice_us_awb_multisyn_configure
		       us_awb-backoff_rules
		       us_awb_data_dir
		       (list (list us_awb_base_dirs "utts.data")
			     (list us_awb_base_dirs "utts.pauses")))

(define (voice_us_awb_multisyn_configure voice)
  "(voice_awb_multisyn_configure voice)
 Set up the current voice to be male US English (Awb) for
 the multisyn unitselection engine."
    (voice_reset)
    (Parameter.set 'Language 'americanenglish)
    (Parameter.set 'PhoneSet 'radio_multisyn)
    (PhoneSet.select 'radio_multisyn)
    (set! token_to_words english_token_to_words)
    (set! pos_lex_name "english_poslex")
    (set! pos_ngram_name 'english_pos_ngram)
    (set! pos_supported t)
    (set! guess_pos english_guess_pos)
    (setup_cmu_lex)
    (lex.select 'cmu)
    (set! postlex_rules_hooks (list postlex_apos_s_check 
				    ))
    (set! postlex_vowel_reduce_table nil)
    (Parameter.set 'Word_Method Unilex_Word)
    (Parameter.set 'Phrase_Method 'cart_tree)
    (set! phrase_cart_tree simple_phrase_cart_tree)
    (Parameter.set 'Pause_Method MultiSyn_Pauses)
    (Parameter.set 'Int_Method nil)
    (Parameter.set 'Int_Target_Method nil)
    (Parameter.set 'Duration_Method nil)
    (Param.set 'Synth_Method 'MultiSyn)
    (Param.set 'unisyn.window_symmetric 0)
)
    
(proclaim_voice
 'cstr_us_awb_arctic_multisyn
 '((language english)(gender male)(dialect american)
   (description "Awb multisyn unitsel voice (default configuration).")))


(provide 'cstr_us_awb_arctic_multisyn)



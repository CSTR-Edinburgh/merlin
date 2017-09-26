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
;;; fls Multisyn Hybrid Voice definition for use with Combilex
;;;
;;; Rob Clark June 2015


;; SCHEME PATHS
(defvar multisyn_lib_dir (path-append libdir "multisyn/"))
(defvar fls_crpx_multisyn_hybrid_dir (cdr (assoc 'cstr_crpx_fls_multisyn_hybridTrim_LSTMParams voice-locations)))
(defvar fls_crpx_data_dir (path-append fls_crpx_multisyn_hybrid_dir "fls"))

(set! load-path (cons (path-append fls_crpx_multisyn_hybrid_dir "festvox/") 
                      load-path))

;; These may or may not be already loaded.
(if (not (member_string multisyn_lib_dir libdir))
    (set! load-path (cons multisyn_lib_dir load-path)))
(if (not (member_string 'combilex-rpx (lex.list)))
    (load (path-append lexdir "combilex/" (string-append 'combilex-rpx ".scm"))))

;; REQUIRES
(require 'hts)
(require 'hts_feats_fls_hybrid)
(require_module 'hts_engine)
(require 'multisyn)
(require 'multisyn_hybrid)
(require 'combilex_phones)
(require 'phrase)
(require 'pos)
;(require 'hts_pauses)

(require 'voice_definition_files/hybrid_synthesis/synth_Blizzard_utts_dot_data)

(set! fls::hts_feats_list
      (load (path-append fls_crpx_multisyn_hybrid_dir "festvox/hts_feats_eng.list") t))

(define (fls::set_hts_engine_params_with_values dur pitch voicing alpha postfilter)
    (set! fls::hts_engine_params
        (list
        (list "-dm1" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/mcep_d1.win"))
        (list "-dm2" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/mcep_d2.win"))
        (list "-df1" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/logF0_d1.win"))
        (list "-df2" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/logF0_d2.win"))
        (list "-da1" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/bndap_d1.win"))
        (list "-da2" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/bndap_d2.win"))
        (list "-td" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/tree-duration.inf"))
        (list "-tm" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/tree-mcep.inf"))
        (list "-tf" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/tree-logF0.inf"))
        (list "-ta" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/tree-bndap.inf"))
        (list "-md" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/duration.pdf"))
        (list "-mm" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/mcep.pdf"))
        (list "-mf" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/logF0.pdf"))
        (list "-ma" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/bndap.pdf"))
        (list "-gm" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/gv-mcep.pdf"))
        (list "-gf" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/gv-lf0.pdf"))
        (list "-ga" (path-append fls_crpx_multisyn_hybrid_dir "hts_engine/gv-bndap.pdf"))
        (list "-a" alpha)
        (list "-b" postfilter)
        '("-s"  48000.0)
        '("-p"  240.0)
        '("-e"  -1.000)
        '("-j"  0.900)
        (list "-q" dur) ; duration_stretch
        ;;'("-fs" 1.500000)             
        (list "-fm" pitch)
        (list "-u"  voicing)
        ;;       '("-l"  0.000000)
        ))
      (set! hts_engine_params fls::hts_engine_params))


(define (fls::voice_reset)
  "(fls::voice_reset)
Reset global variables back to previous voice."
  (HTS_set_null)
  t)


(set! fls-backoff_rules
'(
(l! l)
(n! n)
(E@ e)
(n @)
(A @)
(ae @)
(I @)
(eI @)
(Er @)
(a @)
(E@ @)
(E @)
(O @)
(@Ur @)
(aU @)
(Q @)
(OI @)
(V @)
(U @)
(i@ @)
(U@ @)
(W w )
(s z)
(_ #)

))

;; DATA PATHS

;; Location of Base utterances
(set! fls_crpx_base_dirs (list (path-append fls_crpx_data_dir "utt/")
			   (path-append fls_crpx_data_dir "wav/")
			   (path-append fls_crpx_data_dir "pm/")
			   (path-append fls_crpx_data_dir "coef/")
			   ".utt" ".wav" ".pm" ".coef" ".tcoef"))

(make_voice_definition 'cstr_crpx_fls_multisyn_hybridTrim_LSTMParams 
		       48000
		       'voice_fls_crpx_multisyn_hybrid_configure
		       fls-backoff_rules ;;combilex-rpx-backoff_rules
		       fls_crpx_data_dir
		       (list (list fls_crpx_base_dirs "utts.data")
			     (list fls_crpx_base_dirs "fls_pauses.data")))

(define (voice_fls_crpx_multisyn_hybrid_configure_pre voice)
  "(voice_fls_crpx_multisyn_configure_pre voice)
 Preconfiguration for female British English (fls) for
 the multisyn unitselection engine with Combilex."
  (voice_reset)
  (Parameter.set 'Language 'britishenglish)
  (Param.set 'Ignore_Bad_Phones t) ;; has no effect; but see voice.init in
                                   ;; festival/lib/multisyn/multisyn.scm
  (combilex::select_phoneset))


(define (voice_fls_crpx_multisyn_hybrid_configure voice)
  "(voice_fls_crpx_multisyn_configure voice)
 Set up the current voice to be female British English (fls) for
 the multisyn unitselection engine with Combilex."
  (let (this_voice)
    (voice_reset)
    (Parameter.set 'Language 'britishenglish)
    (combilex::select_phoneset)
    (set! token_to_words english_token_to_words)
    (set! pos_lex_name "english_poslex")
    (set! pos_ngram_name 'english_pos_ngram)
    (set! pos_supported t)
    (set! guess_pos english_guess_pos)
    (lex.select 'combilex-rpx)
    (set! postlex_rules_hooks (list postlex_apos_s_check 
                                    postlex_the_vs_thee
                                    postlex_intervoc_r
                                    postlex_a))
    (set! postlex_vowel_reduce_table nil)
    (Parameter.set 'Word_Method Unilex_Word)
    ;; If you want punctuation to specify phrasing:
    (Parameter.set 'Phrase_Method 'cart_tree)
    (set! phrase_cart_tree simple_phrase_cart_tree)
    ;; If you want probabalistic phrasing prediction
    ;(Parameter.set 'Phrase_Method 'prob_models)
    ;(set! phr_break_params english_phr_break_params)
    (Parameter.set 'Pause_Method MultiSyn_Pauses)
    
    (set! int_tone_cart_tree f2b_int_tone_cart_tree)
    (set! int_accent_cart_tree f2b_int_accent_cart_tree)
    (Parameter.set 'Int_Method Intonation_Tree)
    (Parameter.set 'Int_Target_Method nil)

    (Parameter.set 'Duration_Method nil)

    (fls::set_hts_engine_params_with_values 1.0 0.0 0.5 0.77 0.05)
    (set! hts_feats_list fls::hts_feats_list)
    (set! hts_feats_output fls::hts_feats_output)

    (Param.set 'Synth_Method 'MultiSyn_Hybrid)
    (Param.set 'unisyn.window_symmetric 0))
    (du_voice.setTargetCost voice 'hybrid)

    ; weight the join and target costs
    ;;(du_voice.set_target_cost_weight voice 1)
    ;;(du_voice.set_target_cost_weight voice 0.1)
    ;;(du_voice.set_target_cost_weight voice 0.01)
    ;;(du_voice.set_target_cost_weight voice 0.001)
    ;;(du_voice.set_target_cost_weight voice 0.0005)
    ;;(du_voice.set_target_cost_weight voice 0.0003)
    (du_voice.set_target_cost_weight voice 0.0001)
    ;;(du_voice.set_target_cost_weight voice 0.00005)
    ;;(du_voice.set_target_cost_weight voice 0.00001)
    ;;(du_voice.set_target_cost_weight voice 0.000001)
    ;;(du_voice.set_target_cost_weight voice 0.0000001)
    ;;(du_voice.set_target_cost_weight voice 0.0)
    ;;(du_voice.set_target_cost_weight voice 0.00000001)

    ;;(HTS_set_voice current-voice)
    ;;(set! current_voice_reset fls::voice_reset)
    ;;(set! current-voice "fls_hyprid-rpx") ;;typo? (TOM MERRITT)
    ;;(set! current-voice "fls_hybrid-rpx")
)


(proclaim_voice
 'cstr_crpx_fls_multisyn_hybridTrim_LSTMParams
 '((language english)(gender female)(dialect british)
   (description "fls multisyn hybrid unitsel voice  (Combilex configuration).")))

(provide 'cstr_crpx_fls_multisyn_hybridTrim_LSTMParams)

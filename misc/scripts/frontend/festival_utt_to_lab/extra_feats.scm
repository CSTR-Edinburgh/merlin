;;  ---------------------------------------------------------------  ;;
;;           The HMM-Based Speech Synthesis System (HTS)             ;;
;;                       HTS Working Group                           ;;
;;                                                                   ;;
;;                  Department of Computer Science                   ;;
;;                  Nagoya Institute of Technology                   ;;
;;                               and                                 ;;
;;   Interdisciplinary Graduate School of Science and Engineering    ;;
;;                  Tokyo Institute of Technology                    ;;
;;                     Copyright (c) 2001-2007                       ;;
;;                       All Rights Reserved.                        ;;
;;                                                                   ;;
;;  Permission is hereby granted, free of charge, to use and         ;;
;;  distribute this software and its documentation without           ;;
;;  restriction, including without limitation the rights to use,     ;;
;;  copy, modify, merge, publish, distribute, sublicense, and/or     ;;
;;  sell copies of this work, and to permit persons to whom this     ;;
;;  work is furnished to do so, subject to the following conditions: ;;
;;                                                                   ;;
;;    1. The code must retain the above copyright notice, this list  ;;
;;       of conditions and the following disclaimer.                 ;;
;;                                                                   ;;
;;    2. Any modifications must be clearly marked as such.           ;;
;;                                                                   ;;
;;  NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF TECHNOLOGY,  ;;
;;  HTS WORKING GROUP, AND THE CONTRIBUTORS TO THIS WORK DISCLAIM    ;;
;;  ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL       ;;
;;  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ;;
;;  SHALL NAGOYA INSTITUTE OF TECHNOLOGY, TOKYO INSITITUTE OF        ;;
;;  TECHNOLOGY, HTS WORKING GROUP, NOR THE CONTRIBUTORS BE LIABLE    ;;
;;  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY        ;;
;;  DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  ;;
;;  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTUOUS   ;;
;;  ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          ;;
;;  PERFORMANCE OF THIS SOFTWARE.                                    ;;
;;                                                                   ;;
;;  ---------------------------------------------------------------  ;;
;;
;;  Extra features
;;  From Segment items refer by 
;;
;;  R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_syls_in_phrase
;;  R:SylStructure.parent.parent.R:Phrase.parent.lisp_num_words_in_phrase
;;  lisp_total_words
;;  lisp_total_syls
;;  lisp_total_phrases
;;
;;  The last three will act on any item

(define (distance_to_p_content i)
  (let ((c 0) (rc 0 ) (w (item.relation.prev i "Phrase")))
    (while w
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "contentp"))
      (begin
        (set! rc c)
        (set! w nil))
      (set! w (item.prev w)))
      )
    rc))

(define (distance_to_n_content i)
  (let ((c 0) (rc 0) (w (item.relation.next i "Phrase")))
    (while w
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "contentp"))
      (begin
        (set! rc c)
        (set! w nil))
      (set! w (item.next w)))
      )
    rc))

(define (distance_to_p_accent i)
  (let ((c 0) (rc 0 ) (w (item.relation.prev i "Syllable")))
    (while (and w (member_string (item.feat w "syl_break") '("0" "1")))
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "accented"))
      (begin
        (set! rc c)
        (set! w nil))
        (set! w (item.prev w)))
        )
        rc))

(define (distance_to_n_accent i)
  (let ((c 0) (rc 0 ) (w (item.relation.next i "Syllable")))
    (while (and w (member_string (item.feat w "p.syl_break") '("0" "1")))
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "accented"))
      (begin
        (set! rc c)
        (set! w nil))
        (set! w (item.next w)))
        )
        rc))

(define (distance_to_p_stress i)
  (let ((c 0) (rc 0 ) (w (item.relation.prev i "Syllable")))
    (while (and w (member_string (item.feat w "syl_break") '("0" "1")))
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "stress"))
      (begin
        (set! rc c)
        (set! w nil))
        (set! w (item.prev w)))
        )
        rc))

(define (distance_to_n_stress i)
  (let ((c 0) (rc 0 ) (w (item.relation.next i "Syllable")))
    (while (and w (member_string (item.feat w "p.syl_break") '("0" "1")))
      (set! c (+ 1 c))
      (if (string-equal "1" (item.feat w "stress"))
      (begin
        (set! rc c)
        (set! w nil))
        (set! w (item.next w)))
        )
        rc))

(define (num_syls_in_phrase i)
  (apply 
   +
   (mapcar
    (lambda (w)
      (length (item.relation.daughters w 'SylStructure)))
    (item.relation.daughters i 'Phrase))))

(define (num_words_in_phrase i)
  (length (item.relation.daughters i 'Phrase)))

(define (total_words w)
  (length
   (utt.relation.items (item.get_utt w) 'Word)))

(define (total_syls s)
  (length
   (utt.relation.items (item.get_utt s) 'Syllable)))

(define (total_phrases s)
  (length
   (utt.relation_tree (item.get_utt s) 'Phrase)))


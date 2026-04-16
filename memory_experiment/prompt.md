Synthesized prompt:

We currently have a baseline model that generates reasoning traces. The model selects between two functions (F and G) and produces a sequence of operations. In evaluation, we see that operation-level metrics (like operation accuracy and operation selection) are high, but complete solution accuracy is significantly lower (around 64%). This suggests that while individual steps may look correct, errors accumulate across the chain.

Right now, the approach is implemented as a two-step pipeline: first generate a full trace, then detect where an error occurred using a classifier, and finally inject a correction (a value vector V) at the detected position. However, this is not the intended design.

The correct approach should not separate detection and correction into distinct steps. Instead, the model should be trained as a single integrated system where, at every token generation step, it attends to an external memory. This memory provides value vectors that can correct the model’s behavior. The key idea is that the model learns when to use this memory: attention over memory should be near zero for correct steps and non-zero at steps where corrections are needed. This makes error detection implicit and part of the generation process itself.

Another issue is with the current dataset and ground truth. Right now, the model is effectively learning to choose between F and G in a 50-50 setup, which makes the task ambiguous and difficult. The model is not truly learning which function is correct, only that it must pick one of the two. To make learning feasible, the fine-tuning setup should simplify the domain, for example by ensuring that only one function (e.g. F) is correct in the dataset. This reduces complexity and allows the model to learn consistent behavior.

Conceptually, pretraining operates in a highly diverse and chaotic setting where the model cannot fully infer underlying mechanisms and instead learns statistical correlations. Fine-tuning should restrict the domain so that the model can reliably associate inputs with correct decisions.

To move forward:

Redefine evaluation metrics so that operation selection truly measures correctness, not just selection between F and G.
Remove the explicit detect-then-fix pipeline.
Implement token-level memory attention within the model.
Train the model so that it learns when to apply corrections via memory attention.
Simplify the dataset so that the correct behavior is consistent and learnable.
Investigate why complete solution accuracy is much lower than step-level metrics, likely due to compounding errors.

The goal is to move from a system that generates outputs and then patches errors, to a model that integrates correction into the generation process and becomes inherently self-correcting.

Original transcript:

Já nevím, když to zoptimalizuješ, aby tam to chybu neudělala, jestli návodu to nerozbije, ty ostatní věci, jestli si myslíš, že to nemělo optimalizovat. Jak to udělat, aby se to...
Já jsem to pustil a moc to.
Nefunguje. No a co teď měříš? Teď.
Měřím Operation, akorát si Operation Selection a jestli je výstup stejný. Výstup vypadá, že je stejný,.
Ale.
Ještě musím se podělat, jestli ty metriky jsou správně, ale tady to vypadá, že to je horší. Tady je baseline prostě. Ten přetrenovaný model vygeneruje ty traci a potom tam, kde se detekovala chyba, tak tam se dělá ten proces s tím večkem.
Ale počkej, ty tam máš dva kroky, že detekuješ chybu a pak tam injectuješ večko? A detekuješ to jako jak, tou klasifikaci, jo? Jo, jo. Jo, ale to jako tak na konec to nemůže být. Na konec to je myšleno tak, že ty tam jako nemáš nějaký separátní detekovací krok. Ty tam středka děláš v každém kroku, u každého touknu děláš tu attention na tu memory. Akorát chceš, aby ta attention byla nulová všude, kromě toho, kroků, kde to má se obravit. Takže ty máš ten model a ten model vygeneruje nějaký chain a v určitém kroku tam máš chybu.
Tady je to v 17 případech ze 128 exámpů.
Jakože vybral tu špatnou funkci, jo? S těch dvou? No, no, jo. A to je ta baseline, jo? To je ta baseline. Že vlastně to ještě není jako... Že to ještě není jako napainturováno vůbec. Takže on... Tam měříš operation accuracy, to znamená...
Jestli jsou ty funkce správně spočítané.
Jestli vybírat ty správné funkce?
Jestli jsou správně spočítané.
To je tak, operation selection je ta druhá. Jasné, takže... Tak to je celkem přesné, ten model, takže on ví tu správnou funkci.
No, to je dobrá otázka. Já se na to musím ještě podívat. To teďka doběhlo, než jsme se zavolali. Takže já to ještě si s tím hraju. Ale přijde mi to divné, že tady jako ten Complete Solution, kde všechno je správně, včetně výsledků, výsledek je stejný jako ten prompt. Takže to je jako 64%, když to tady ty částečné jsou 98 a 95.
A ty Ground Truths, ty jsou dělané tak, že vybíráš vždycky to F nebo ty sedlejí jak? No, tak ten.
Ground Truths, jako tady v tomhle případě, u té baseliny, tak tady je to buď F nebo G, že jo, tady.
Jako... Takže tam se pořád jako náhodně vybírá?
Jo, tak tady jako... Ten model, jako tam je 50-50 prostě. A tady jsem řekl prostě, že když tam udělá chybu, tak jenom ať vymění písmenou. Tam jako jiný ground truth asi není úplně. No ale říkám, ještě to musím na to mrknout, já bych chtěl probrat ten proces, ale ty teda říkáš, že to nebude na dva kroky, že to bude na jeden, že tam bude ten attention na tu mapu, attention.
Na tu memory. Tam bude jeden krok, tam bude, kde se detekuje vlastně, kde byla ta chyba. Jo, to tam jako bude, ten explicitní detekování, kde chcem teda to změnit. Ale pak to učení a ten model, jak bude fungovat, on nebude fungovat, že by to nejdříve detekoval. On si vlastně bude v každém touknu, ten nejako nafantinovaný model, ten, co bude mít tu paměť vzaintegrovanou, tak on bude v každém touknu dělat attention na tu memory a bude agregovat ten value vector, akorátže právě v každém kroku, kromě těch, kde udal tu chybu, by to mělo mít jako Takže... Já uvažuji, jak to... A teďka ty trací jsou jak dlouhé?
300 Touknut,.
Třeba. A jestli to je až těch pět operací, nebo jestli to je kratší?
No, tak jako... Ty jsi to měl,.
Myslím, že 3 až 5, ne? Původně.
No, 3 až 5 to je furt stejné. Akorát vektor je delky.
6. Tak to je zvláštní, že on to fakt tak dobře predikovat ty funkce. Jakože v tom baselineu, já jsem.
Si myslel, že teda... No, tak to víme od začátku, že to , že se vybírá efektivní údejčko. To celou dobu vybírat.
No to jo, ale já jsem měl pocit, že on sice ví, že má mít buď F-ku nebo G-čku, ale že neví které z toho. No.
To jo. A tady tohle číslo ti neříká, že vybral tu správné. Říká, že vybral z F-ka nebo z G-čka.
Jo tak, jasné. Tak to jsem si myslel, že právě... Že by se právě jako... Že by se právě to omezilo, že ty Ground Truths vybyly jenom na to F, protože ono to je asi moc těžké, pro něho si je to propočítat a zjistit, jestli má popoužívat to G nebo F, ale že potom ten fine tuning by šel v setupu, že vždycky třeba se má vybírat F. Já.
Nevím, jak tohle má fungovat vzhledem k tomu, jaký máš input. Protože v inputu máš ten output.
Vektor. Ty bys musel vytáhnout z example, kde se vždycky vybral jenom F. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. Ano. An Já si to představuju v reálu, že by to bylo tak, že ty máš jazykový model, který si vygeneruje nějaký chain a dojde k nějakému řešení. A teďka tam někde udělá chyby, že tam se použije špatné věci a chainuje si tam špatně ty procedury.
A teďka tam na to se pustí nějaký další jazykový model, který to jako by zčekuje ten reasoning a najde tam, kde tam byla nějaká chyba. Tady seš fakt nerozhodnul, tady jsi měl uvažovat tímto způsobem, a ne takhle. A takže by se to jako detekovalo to místo a tam by se potom udělalo nějaký fine tuning, který by si jako ten model zaintegroval, že tam se musí rozhodnout přiště vždycky takhle. Ono to tak může vlastně být, že řekněme, že jak se to určitě vlastně zdá z toho internetu, tak vlastně ty dáta jsou, To je tak brutálně různoroké, že to víceméně není predikovatelné všechno. Že to vlastně, ten model tam musí jako právě...
Já si představuju tak, že je jako jeden extrém, že on jede čistě statisticky, že on vlastně vůbec nechápe ten mechanismus a jenom zkladka to dává jako nějakou statistickou asociaci si tam vytvoří. V takovémhle kontextu vždycky volím to nebo to, ale vůbec nechápe proč. Nechápe ten mechanismus zatím. Zatímco na druhém konci je to, že ten model fakt dokáže úplně pochopit. Chápe ten mechanismus a dokáže nad ním uvažovat a dokáže si to celé simulovat. A ty právě jak to učíš, jak se dělá ten pre-training, tak tam je to takový chaos, že on nemá šanci si to pochopit všechno, že by si vytvrdl ten mechanizmus v sobě. Ale učíš si tam nějaké statistické korolace. A pak jak se dělá ten fine-tuning na nějaký konkrétní setup, tak tam je to třeba mnohem jednodušší.
A on si z těch statistických korolací musí vytáhnout ty správné varianty. Jakmile to je v té konkrétní doméně, tak je už je tam najednou lehčí. A on si potřebuje násociovat ty správné věci pro tu konkrétní doménu. U nás ta konkrétní doména znamená, že už se nepoužívá G, už se vždycky používá F a tím pádem je ta úloha mnohem lehčí. Protože on si vlastně... No, jo, tak to je. Při tom retainingu to je zhradka chaos. Ono to teoreticky je predipovatelné, ale je to moc složité. Protože on by si musel nasimulovat ty funkce a zjistit, které se měli použít. A pak až se rozhodnout, kterou by si použil F2 nebo G, což ten model neumí.
Ale u toho fine-tuningu je to lehké, protože on ví, že vždycky má vybrat F, takže si jenom potřebuje spočítat, která ta funkce podle toho F to je. Jo. Dobré. Tak asi... Dobré, takže jestli to... ...Ještě ještě... ...Si na to ještě podíváš? Jo, pohážu se, no. Ten setup a... ...Můžeme se třeba zase zítra spojit. Dobré, dobré. Jo, super, tak jo, tak zatím. Tak jo, tak tím čau. Čau.
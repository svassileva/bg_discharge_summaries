"""
Bulgarian Medical Abbreviations Dictionary
===========================================
A curated map of abbreviations (Cyrillic and Latin) used in Bulgarian clinical
practice, organised by category.  Each entry maps the abbreviated form to a
tuple of:
    (full_Bulgarian_term, English_translation)

Sources
-------
The dictionary was compiled from the following reference sources:

1. **Bulgarian national clinical pathways (Клинични пътеки)**
   Publisher: Ministry of Health of Bulgaria (Министерство на здравеопазването)
   URL: https://www.mh.government.bg/bg/normativni-aktove/naredbi/klinichni-pateki/
   Notes: Primary source for procedure codes, ward names (КП), and
          diagnosis-specific abbreviations used in state-funded treatment.

2. **National Health Insurance Fund — disease nomenclature (МКБ-10)**
   Publisher: Национална здравноосигурителна каса (НЗОК)
   URL: https://www.nhif.bg/
   Notes: Administrative abbreviations (НЗОК, ТЕЛК, НЕЛК, ЛКК, ОПЛ),
          ICD-10 coding conventions, and reimbursement-related terms.

3. **Bulgarian Drug Agency — approved product labelling (КХП)**
   Publisher: Изпълнителна агенция по лекарствата (ИАЛ)
   URL: https://www.bda.bg/
   Notes: Pharmacotherapy abbreviations (АСЕ-И, АРБ, ББ, НОАК, НСПВС,
          НМХ, ИПП) cross-checked against approved Summary of Product
          Characteristics documents.

4. **УМБАЛ "Александровска" — clinical documentation templates**
   Publisher: УМБАЛ "Александровска" ЕАД, Sofia
   URL: https://www.alexandrovska.com/
   Notes: Discharge-summary header abbreviations, ward/unit designations
          (КАИЛ, ОАИЛ, ИТО, ДКЦ), and investigative short-forms (ЕхоКС,
          ДС, КАГ) as used in real Bulgarian hospital records.

5. **Cardiology Society of Bulgaria — clinical guidelines**
   Publisher: Българско дружество по кардиология
   URL: https://bdkardio.org/
   Notes: Cardiovascular abbreviations (АХ, ИБС, СН, МИ, АКС, ПМ, ДКМП,
          ХКМП, ПЕРК/ПКИ, АКБ) aligned with ESC guideline translations.

6. **Bulgarian Neurology Society — stroke guidelines**
   Publisher: Българско дружество по неврология
   Notes: Stroke and cerebrovascular abbreviations (ИМИ, ХМИ, ВМК, ТИА,
          ЦВБ, САК, СМА, ВСА) following national stroke protocol terminology.

7. **Bulgarian Endocrinology Society — diabetes & thyroid guidelines**
   Publisher: Българско дружество по ендокринология
   URL: https://bde.bg/
   Notes: Endocrine abbreviations (ЗД, ЗД1, ЗД2, ДКА, ХГС, ЩЖ, ХТ,
          ТСХ/TSH, HbA1c, ПТХ) drawn from national diabetes and
          thyroid-disease management guidelines.

8. **"Наръчник по медицинска терминология" (Medical Terminology Handbook)**
   Authors: Проф. д-р Ив. Миланов и колектив
   Publisher: Издателство "Медицина и физкултура", Sofia
   Notes: Reference for anatomical (ЦНС, ПНС, ЛК, ДК, ЛП, ДП) and
          general clinical abbreviations standardised in Bulgarian
          academic medicine.

9. **"Вътрешни болести" (Internal Medicine) — Bulgarian textbook**
   Authors: Проф. д-р Ат. Кумчев и колектив
   Publisher: Издателство "Арсо", Sofia
   Notes: Internal-medicine abbreviations (ХОББ, БА, ЖКБ, ЯБ, ЦД,
          ВХ, ОП, ХП, РА, ОА, СЛЕ) as taught in Bulgarian medical schools.

10. **ICD-10-CM / MKБ-10 bilateral mapping**
    Publisher: WHO / НЗОК Bulgarian adaptation
    Notes: Used to cross-validate disease-name abbreviations against the
           official Bulgarian ICD-10 disease classification codes to ensure
           consistency with coded hospital discharge data.
"""

# ---------------------------------------------------------------------------
# Core dictionary
# ---------------------------------------------------------------------------

BULGARIAN_MEDICAL_ABBREVIATIONS: dict[str, tuple[str, str]] = {

    # -------------------------------------------------------------------
    # Diagnoses / Conditions
    # -------------------------------------------------------------------
    "АХ":    ("Артериална хипертония", "Arterial hypertension"),
    "АГ":    ("Артериална хипертония", "Arterial hypertension"),
    "ИБС":   ("Исхемична болест на сърцето", "Ischaemic heart disease"),
    "ИБC":   ("Исхемична болест на сърцето", "Ischaemic heart disease"),
    "МИ":    ("Миокарден инфаркт", "Myocardial infarction"),
    "АМИ":   ("Остър миокарден инфаркт", "Acute myocardial infarction"),
    "STEMI": ("Инфаркт на миокарда с елевация на ST сегмент", "ST-elevation myocardial infarction"),
    "NSTEMI":("Инфаркт на миокарда без елевация на ST сегмент", "Non-ST-elevation myocardial infarction"),
    "СН":    ("Сърдечна недостатъчност", "Heart failure"),
    "ХСН":   ("Хронична сърдечна недостатъчност", "Chronic heart failure"),
    "ОСН":   ("Остра сърдечна недостатъчност", "Acute heart failure"),
    "ПМ":    ("Предсърдно мъждене", "Atrial fibrillation"),
    "ФП":    ("Фибрилация на предсърдията", "Atrial fibrillation"),
    "ПТ":    ("Предсърдно трептене", "Atrial flutter"),
    "ПНД":   ("Пароксизмална надкамерна тахикардия", "Paroxysmal supraventricular tachycardia"),
    "КТ":    ("Камерна тахикардия", "Ventricular tachycardia"),          # context-dependent (also 'Компютърна томография')
    "КМ":    ("Кардиомиопатия", "Cardiomyopathy"),
    "ДКМП":  ("Дилатационна кардиомиопатия", "Dilated cardiomyopathy"),
    "ХКМП":  ("Хипертрофична кардиомиопатия", "Hypertrophic cardiomyopathy"),
    "АС":    ("Аортна стеноза", "Aortic stenosis"),
    "АР":    ("Аортна регургитация", "Aortic regurgitation"),
    "МС":    ("Митрална стеноза", "Mitral stenosis"),
    "МР":    ("Митрална регургитация", "Mitral regurgitation"),
    "ТР":    ("Трикуспидална регургитация", "Tricuspid regurgitation"),
    "ХОББ":  ("Хронична обструктивна белодробна болест", "Chronic obstructive pulmonary disease"),
    "ХОББ":  ("Хронична обструктивна белодробна болест", "COPD"),
    "БА":    ("Бронхиална астма", "Bronchial asthma"),
    "ДН":    ("Дихателна недостатъчност", "Respiratory failure"),
    "ТЕЛА":  ("Тромбоемболия на белодробната артерия", "Pulmonary embolism"),
    "ТГВ":   ("Тромбоза на дълбоките вени", "Deep vein thrombosis"),
    "ДВТ":   ("Дълбока венозна тромбоза", "Deep vein thrombosis"),
    "ЗД":    ("Захарен диабет", "Diabetes mellitus"),
    "ДМ":    ("Диабетус мелитус / Захарен диабет", "Diabetes mellitus"),
    "ЗД1":   ("Захарен диабет тип 1", "Type 1 diabetes mellitus"),
    "ЗД2":   ("Захарен диабет тип 2", "Type 2 diabetes mellitus"),
    "ДМ тип 1": ("Захарен диабет тип 1", "Type 1 diabetes mellitus"),
    "ДМ тип 2": ("Захарен диабет тип 2", "Type 2 diabetes mellitus"),
    "ХБЗ":   ("Хронично бъбречно заболяване", "Chronic kidney disease"),
    "ХБН":   ("Хронична бъбречна недостатъчност", "Chronic renal failure"),
    "ОБЗ":   ("Остро бъбречно заболяване", "Acute kidney injury"),
    "ОБН":   ("Остра бъбречна недостатъчност", "Acute renal failure"),
    "ЦВБ":   ("Цереброваскуларна болест", "Cerebrovascular disease"),
    "МСБ":   ("Мозъчносъдова болест", "Cerebrovascular disease"),
    "ИМИ":   ("Исхемичен мозъчен инсулт", "Ischaemic stroke"),
    "ХМИ":   ("Хеморагичен мозъчен инсулт", "Haemorrhagic stroke"),
    "ВМК":   ("Вътремозъчен кръвоизлив", "Intracerebral haemorrhage"),
    "САК":   ("Субарахноидален кръвоизлив", "Subarachnoid haemorrhage"),
    "ТИА":   ("Транзиторна исхемична атака", "Transient ischaemic attack"),
    "ХТТТТ": ("Хипотиреоидизъм", "Hypothyroidism"),
    "ХТ":    ("Хипотиреоидизъм", "Hypothyroidism"),
    "ХперТ": ("Хипертиреоидизъм", "Hyperthyroidism"),
    "ГПЖ":   ("Гастроезофагеална рефлуксна болест", "Gastroesophageal reflux disease"),
    "ЯБ":    ("Язвена болест", "Peptic ulcer disease"),
    "ДЯБ":   ("Дуоденална язвена болест", "Duodenal ulcer"),
    "ЖКК":   ("Жълтеница / Жлъчнокаменна болест", "Cholelithiasis"),
    "ЖКБ":   ("Жлъчнокаменна болест", "Cholelithiasis"),
    "БЗЧ":   ("Болест на Задния черен дроб / Цироза", "Liver cirrhosis"),
    "ЧД":    ("Черен дроб", "Liver"),
    "ЦД":    ("Цироза на черен дроб", "Liver cirrhosis"),
    "ВХ":    ("Вирусен хепатит", "Viral hepatitis"),
    "ОП":    ("Остър панкреатит", "Acute pancreatitis"),
    "ХП":    ("Хроничен панкреатит", "Chronic pancreatitis"),
    "СПИН":  ("Синдром на придобита имунна недостатъчност", "AIDS"),
    "ХИВ":   ("Вирус на хумания имунодефицит", "HIV"),
    "ТБК":   ("Туберкулоза", "Tuberculosis"),
    "ТБ":    ("Туберкулоза", "Tuberculosis"),
    "РА":    ("Ревматоиден артрит", "Rheumatoid arthritis"),
    "ОА":    ("Остеоартрит", "Osteoarthritis"),
    "СЛЕ":   ("Системен лупус еритематозус", "Systemic lupus erythematosus"),
    "СЛЕ":   ("Системен лупус еритематозус", "SLE"),
    "ОС":    ("Остеопороза", "Osteoporosis"),
    "ИПН":   ("Инфекция на пикочните пътища", "Urinary tract infection"),
    "УТИ":   ("Уринарна трактова инфекция", "Urinary tract infection"),
    "ПН":    ("Пневмония", "Pneumonia"),
    "БП":    ("Бронхопневмония", "Bronchopneumonia"),
    "ИСТИ":  ("Инфекция на страните на тялото", "Soft tissue infection"),
    "ГК":    ("Гръдна клетка", "Chest / Thorax"),
    "ОГК":   ("Органи на гръдна клетка", "Thoracic organs"),
    "АКС":   ("Остър коронарен синдром", "Acute coronary syndrome"),
    "НАП":   ("Нестабилна ангина пекторис", "Unstable angina"),
    "САП":   ("Стабилна ангина пекторис", "Stable angina"),

    # -------------------------------------------------------------------
    # Diagnostic investigations
    # -------------------------------------------------------------------
    "ЕКГ":   ("Електрокардиограма", "Electrocardiogram"),
    "ЕМГ":   ("Електромиография", "Electromyography"),
    "ЕЕГ":   ("Електроенцефалограма", "Electroencephalogram"),
    "ЕхоКС": ("Ехокардиоскопия / Ехокардиография", "Echocardiography"),
    "ЕхоКГ": ("Ехокардиография", "Echocardiography"),
    "ЯМР":   ("Ядрено-магнитен резонанс", "Magnetic resonance imaging"),
    "МРТ":   ("Магнитно-резонансна томография", "Magnetic resonance imaging"),
    "МРА":   ("Магнитно-резонансна ангиография", "MR angiography"),
    "СКТ":   ("Спирална компютърна томография", "CT scan"),
    "МСКТ":  ("Мултисрезова компютърна томография", "Multi-slice CT"),
    "КТА":   ("Компютърно-томографска ангиография", "CT angiography"),
    "КАГ":   ("Коронарна ангиография", "Coronary angiography"),
    "АГ":    ("Ангиография", "Angiography"),                              # context-dependent
    "УЗИ":   ("Ултразвуково изследване", "Ultrasound examination"),
    "УЗСС":  ("Ултразвуково изследване на сърдечносъдовата система", "Cardiovascular ultrasound"),
    "ДС":    ("Дуплекссканиране", "Duplex scan"),
    "ОАК":   ("Обща анализ на кръвта / Обща кръвна картина", "Complete blood count"),
    "ОКК":   ("Обща кръвна картина", "Complete blood count"),
    "ОАМ":   ("Обща анализ на урина", "Urinalysis"),
    "БАК":   ("Биохимичен анализ на кръвта", "Serum biochemistry"),
    "КГА":   ("Кръвно-газов анализ", "Blood gas analysis"),
    "КГА":   ("Кръвно-газов анализ", "Arterial blood gas"),
    "ЛП":    ("Лумбална пункция", "Lumbar puncture"),
    "СМП":   ("Спинномозъчна пункция", "Cerebrospinal fluid analysis"),
    "РЕНТГ": ("Рентгенография", "X-ray"),
    "РЕТ":   ("Ретгенография", "Radiograph"),
    "СЦГ":   ("Сцинтиграфия", "Scintigraphy"),

    # -------------------------------------------------------------------
    # Vital signs & clinical parameters
    # -------------------------------------------------------------------
    "АН":    ("Артериално налягане", "Arterial blood pressure"),
    "КН":    ("Кръвно налягане", "Blood pressure"),
    "АКН":   ("Артериално кръвно налягане", "Arterial blood pressure"),
    "СН":    ("Систолно налягане", "Systolic pressure"),               # context-dependent
    "ДН":    ("Диастолно налягане", "Diastolic pressure"),             # context-dependent
    "ЧСС":   ("Честота на сърдечните съкращения", "Heart rate"),
    "ЧД":    ("Честота на дишане", "Respiratory rate"),
    "ТТ":    ("Телесна температура", "Body temperature"),
    "ТТ":    ("Телесна температура", "Body temperature"),
    "SpO2":  ("Сатурация на артериалната кръв с кислород", "Oxygen saturation"),
    "SaO2":  ("Сатурация на артериалната кръв с кислород", "Arterial oxygen saturation"),
    "ТМ":    ("Телесна маса", "Body weight"),
    "БМИ":   ("Индекс на телесна маса", "Body mass index"),
    "ИТМ":   ("Индекс на телесна маса", "Body mass index"),
    "GCS":   ("Скала на кома по Глазгоу", "Glasgow Coma Scale"),
    "ФИ":    ("Фракция на изтласкване", "Ejection fraction"),
    "ФИ ЛК": ("Фракция на изтласкване на лява камера", "Left ventricular ejection fraction"),
    "LVEF":  ("Фракция на изтласкване на лявата камера", "Left ventricular ejection fraction"),
    "ЦВН":   ("Централно венозно налягане", "Central venous pressure"),
    "РФ":    ("Ритмична / аритмична функция", "Cardiac rhythm"),

    # -------------------------------------------------------------------
    # Pharmacotherapy
    # -------------------------------------------------------------------
    "АСА":   ("Ацетилсалицилова киселина", "Acetylsalicylic acid / Aspirin"),
    "АСЕ-И": ("АСЕ инхибитори", "ACE inhibitors"),
    "АПФ-И": ("Инхибитори на ангиотензин-превръщащия фермент", "ACE inhibitors"),
    "АРБ":   ("Ангиотензин II рецепторни блокери", "Angiotensin II receptor blockers"),
    "СРБ":   ("Сердечно-разширяващи блокери / ССБ", "Calcium channel blockers"),
    "ССБ":   ("Блокери на калциевите канали", "Calcium channel blockers"),
    "ББ":    ("Бета-блокери", "Beta-blockers"),
    "ДМ":    ("Диуретици", "Diuretics"),
    "НМХ":   ("Нискомолекулярен хепарин", "Low molecular weight heparin"),
    "НФХ":   ("Нефракциониран хепарин", "Unfractionated heparin"),
    "ОАК":   ("Орален антикоагулант", "Oral anticoagulant"),
    "НОАК":  ("Нов орален антикоагулант", "Novel oral anticoagulant / DOAC"),
    "ДОАК":  ("Директен орален антикоагулант", "Direct oral anticoagulant"),
    "НСПВС": ("Нестероидни противовъзпалителни средства", "NSAIDs"),
    "НПВС":  ("Нестероидни противовъзпалителни средства", "NSAIDs"),
    "КС":    ("Кортикостероиди", "Corticosteroids"),
    "АБ":    ("Антибиотик", "Antibiotic"),
    "ИПП":   ("Инхибитори на протонната помпа", "Proton pump inhibitors"),
    "СТ":    ("Статини", "Statins"),
    "МАОИ":  ("Инхибитори на моноаминооксидазата", "MAO inhibitors"),
    "ССРИ":  ("Селективни инхибитори за обратното захващане на серотонина", "SSRIs"),

    # -------------------------------------------------------------------
    # Laboratory values
    # -------------------------------------------------------------------
    "Hb":    ("Хемоглобин", "Haemoglobin"),
    "Ht":    ("Хематокрит", "Haematocrit"),
    "Ер":    ("Еритроцити", "Red blood cells"),
    "Лк":    ("Левкоцити", "White blood cells"),
    "Тр":    ("Тромбоцити", "Platelets"),
    "СУЕ":   ("Скорост на утаяване на еритроцитите", "Erythrocyte sedimentation rate"),
    "СРП":   ("С-реактивен протеин", "C-reactive protein"),
    "CRP":   ("С-реактивен протеин", "C-reactive protein"),
    "ПТ":    ("Протромбиново време", "Prothrombin time"),             # context-dependent
    "АПТТ":  ("Активирано парциално тромбопластиново време", "APTT"),
    "INR":   ("Международно нормализирано отношение", "International normalised ratio"),
    "МНО":   ("Международно нормализирано отношение", "International normalised ratio"),
    "ГКК":   ("Глюкоза в кръвта на гладно", "Fasting blood glucose"),
    "ГКГ":   ("Гликиран хемоглобин", "Glycated haemoglobin"),
    "HbA1c": ("Гликиран хемоглобин", "Glycated haemoglobin / HbA1c"),
    "ТГ":    ("Триглицериди", "Triglycerides"),
    "ОХ":    ("Общ холестерол", "Total cholesterol"),
    "ЛДЛ":   ("Липопротеин с ниска плътност", "LDL cholesterol"),
    "ХДЛ":   ("Липопротеин с висока плътност", "HDL cholesterol"),
    "LDL":   ("Липопротеин с ниска плътност", "LDL cholesterol"),
    "HDL":   ("Липопротеин с висока плътност", "HDL cholesterol"),
    "Кр":    ("Серумен креатинин", "Serum creatinine"),
    "СГФ":   ("Скорост на гломерулна филтрация", "Glomerular filtration rate"),
    "eGFR":  ("Изчислена скорост на гломерулна филтрация", "Estimated GFR"),
    "Ур":    ("Серумна урея", "Serum urea"),
    "Ал":    ("Серумен албумин", "Serum albumin"),
    "БИ":    ("Билирубин", "Bilirubin"),
    "АЛТ":   ("Аланин аминотрансфераза", "Alanine aminotransferase"),
    "АСТ":   ("Аспартат аминотрансфераза", "Aspartate aminotransferase"),
    "ГГТ":   ("Гама-глутамил трансфераза", "Gamma-GT"),
    "АФ":    ("Алкална фосфатаза", "Alkaline phosphatase"),
    "ЛДХ":   ("Лактат дехидрогеназа", "Lactate dehydrogenase"),
    "Na":    ("Натрий", "Sodium"),
    "K":     ("Калий", "Potassium"),
    "Ca":    ("Калций", "Calcium"),
    "Mg":    ("Магнезий", "Magnesium"),
    "Cl":    ("Хлор", "Chloride"),
    "pH":    ("Водороден показател", "pH"),
    "pO2":   ("Парциално налягане на кислород", "Partial pressure of oxygen"),
    "pCO2":  ("Парциално налягане на въглероден диоксид", "Partial pressure of CO2"),
    "ТК":    ("Тропонин К", "Troponin"),
    "Тропонин": ("Тропонин", "Troponin"),
    "КФК":   ("Креатин фосфокиназа", "Creatine phosphokinase"),
    "CK":    ("Креатин киназа", "Creatine kinase"),
    "CK-MB": ("МВ изоензим на креатин киназата", "CK-MB"),
    "NT-proBNP": ("N-краен про-мозъчен натриуретичен пептид", "NT-proBNP"),
    "BNP":   ("Мозъчен натриуретичен пептид", "Brain natriuretic peptide"),
    "TSH":   ("Тиреоид-стимулиращ хормон", "Thyroid-stimulating hormone"),
    "FT4":   ("Свободен тироксин", "Free thyroxine"),
    "FT3":   ("Свободен трийодтиронин", "Free triiodothyronine"),
    "PSA":   ("Простат-специфичен антиген", "Prostate-specific antigen"),
    "ПСА":   ("Простат-специфичен антиген", "Prostate-specific antigen"),
    "ПТХ":   ("Паратхормон", "Parathyroid hormone"),
    "PTH":   ("Паратхормон", "Parathyroid hormone"),
    "25-OHВД3": ("25-хидроксивитамин Д3", "25-hydroxyvitamin D3"),
    "ВД":    ("Витамин Д", "Vitamin D"),
    "В12":   ("Витамин В12", "Vitamin B12"),
    "ФК":    ("Фолиева киселина", "Folic acid"),
    "Фер":   ("Феритин", "Ferritin"),

    # -------------------------------------------------------------------
    # Medical procedures
    # -------------------------------------------------------------------
    "ПЕРК":  ("Перкутанна коронарна интервенция", "Percutaneous coronary intervention"),
    "ПКИ":   ("Перкутанна коронарна интервенция", "Percutaneous coronary intervention"),
    "CABG":  ("Аорто-коронарен байпас", "Coronary artery bypass grafting"),
    "АКБ":   ("Аорто-коронарен байпас", "Coronary artery bypass graft"),
    "ТВТ":   ("Трансвенозна временна пейсинг", "Transvenous temporary pacing"),
    "ИКД":   ("Имплантируем кардиовертер-дефибрилатор", "Implantable cardioverter-defibrillator"),
    "ЕФС":   ("Електрофизиологично изследване", "Electrophysiological study"),
    "КПР":   ("Кардио-пулмонарна реанимация", "Cardiopulmonary resuscitation"),
    "СПР":   ("Сърдечно-белодробна реанимация", "Cardiopulmonary resuscitation"),
    "ИВЛ":   ("Изкуствена белодробна вентилация", "Mechanical ventilation"),
    "МВ":    ("Механична вентилация", "Mechanical ventilation"),
    "НИВЛ":  ("Неинвазивна вентилация", "Non-invasive ventilation"),
    "НИПП":  ("Неинвазивно позитивно налягане в дихателните пътища", "Non-invasive positive pressure ventilation"),
    "CPAP":  ("Постоянно позитивно налягане в дихателните пътища", "Continuous positive airway pressure"),
    "BiPAP": ("Двустепенно позитивно налягане в дихателните пътища", "Bilevel positive airway pressure"),
    "ДТ":    ("Дренаж на плеврата", "Pleural drainage"),
    "ЦВК":   ("Централен венозен катетър", "Central venous catheter"),
    "УМК":   ("Уринарен пикочен катетър", "Urinary catheter"),
    "НГТ":   ("Назогастрална тръба", "Nasogastric tube"),
    "ЕЗ":    ("Ендотрахеална занасяне", "Endotracheal intubation"),
    "ОТ":    ("Оперативно лечение", "Surgical treatment"),
    "ОВ":    ("Оперативна венозна интервенция", "Surgical venous intervention"),
    "ОА":    ("Оперативна артериална интервенция", "Surgical arterial intervention"),
    "ТЕП":   ("Тотална ендопротеза", "Total endoprosthesis"),

    # -------------------------------------------------------------------
    # Anatomy
    # -------------------------------------------------------------------
    "ЦНС":   ("Централна нервна система", "Central nervous system"),
    "ПНС":   ("Периферна нервна система", "Peripheral nervous system"),
    "ВНС":   ("Вегетативна нервна система", "Autonomic nervous system"),
    "ГМ":    ("Главен мозък", "Brain"),
    "ГМС":   ("Главно-мозъчен ствол", "Brainstem"),
    "ЧМН":   ("Черепномозъчни нерви", "Cranial nerves"),
    "МК":    ("Малкият мозък", "Cerebellum"),
    "СГМ":   ("Структури на главния мозък", "Brain structures"),
    "ЛК":    ("Лява камера", "Left ventricle"),
    "ДК":    ("Дясна камера", "Right ventricle"),
    "ЛП":    ("Ляво предсърдие", "Left atrium"),
    "ДП":    ("Дясно предсърдие", "Right atrium"),
    "МК":    ("Митрална клапа", "Mitral valve"),               # context-dependent
    "АК":    ("Аортна клапа", "Aortic valve"),
    "ТК":    ("Трикуспидална клапа", "Tricuspid valve"),        # context-dependent
    "БА":    ("Белодробна артерия", "Pulmonary artery"),         # context-dependent
    "БВ":    ("Белодробна вена", "Pulmonary vein"),
    "МСП":   ("Междупредсърдна преграда", "Interatrial septum"),
    "МКП":   ("Междукамерна преграда", "Interventricular septum"),
    "Аорта": ("Аорта", "Aorta"),
    "ГМА":   ("Главна мозъчна артерия", "Major cerebral artery"),
    "СМА":   ("Средна мозъчна артерия", "Middle cerebral artery"),
    "ПМА":   ("Предна мозъчна артерия", "Anterior cerebral artery"),
    "ЗМА":   ("Задна мозъчна артерия", "Posterior cerebral artery"),
    "ВСА":   ("Вътрешна сонна артерия", "Internal carotid artery"),
    "ОСА":   ("Обща сонна артерия", "Common carotid artery"),
    "ВА":    ("Гръбначномозъчна (вертебрална) артерия", "Vertebral artery"),
    "КБА":   ("Каротидно-базиларна артерия", "Carotid-basilar artery"),

    # -------------------------------------------------------------------
    # Healthcare settings & administrative
    # -------------------------------------------------------------------
    "УМБАЛ": ("Университетска многопрофилна болница за активно лечение", "University hospital"),
    "МБАЛ":  ("Многопрофилна болница за активно лечение", "General hospital"),
    "ДКЦ":   ("Диагностично-консултативен център", "Diagnostic and consultative centre"),
    "КАБ":   ("Клиника по анестезиология и реанимация", "Anaesthesiology and intensive care unit"),
    "ИТО":   ("Интензивно терапевтично отделение", "Intensive care unit"),
    "КАИЛ":  ("Клиника по анестезиология и интензивно лечение", "Anaesthesiology and intensive care"),
    "ОАИЛ":  ("Отделение по анестезиология и интензивно лечение", "Anaesthesiology department"),
    "КВО":   ("Кардиологично / вътрешно отделение", "Cardiology / internal medicine ward"),
    "НО":    ("Неврологично отделение", "Neurology ward"),
    "ОПЛ":   ("Общопрактикуващ лекар", "General practitioner"),
    "ЛКК":   ("Лекарска консултативна комисия", "Medical advisory committee"),
    "ТЕЛК":  ("Трудово-експертна лекарска комисия", "Medical labour assessment committee"),
    "НЕЛК":  ("Национална експертна лекарска комисия", "National medical expert committee"),
    "НЗОК":  ("Национална здравна осигурителна каса", "National Health Insurance Fund"),
    "МЗ":    ("Министерство на здравеопазването", "Ministry of Health"),
    "ИЗ":    ("Изписна записка / История на заболяването", "Discharge summary / Medical history"),
    "АФ":    ("Амбулаторен лист / формуляр", "Outpatient referral form"),           # context-dependent
    "ЕПР":   ("Електронно пациентско досие", "Electronic patient record"),
    "ДС":    ("Диагностичен стандарт", "Diagnostic standard"),
    "КП":    ("Клинична пътека", "Clinical pathway"),

    # -------------------------------------------------------------------
    # Specialty / subspecialty names
    # -------------------------------------------------------------------
    "КАР":   ("Кардиология", "Cardiology"),
    "НЕВ":   ("Неврология", "Neurology"),
    "ЕНД":   ("Ендокринология", "Endocrinology"),
    "ГАС":   ("Гастроентерология", "Gastroenterology"),
    "НЕФ":   ("Нефрология", "Nephrology"),
    "РЕВ":   ("Ревматология", "Rheumatology"),
    "ОНК":   ("Онкология", "Oncology"),
    "ХЕМ":   ("Хематология", "Haematology"),
    "ПУЛ":   ("Пулмология", "Pulmonology"),
    "ОРТ":   ("Ортопедия", "Orthopaedics"),
    "ТРМ":   ("Травматология", "Traumatology"),
    "УРО":   ("Урология", "Urology"),
    "ГИН":   ("Гинекология", "Gynaecology"),
    "АКУ":   ("Акушерство", "Obstetrics"),
    "ПЕД":   ("Педиатрия", "Paediatrics"),
    "ОФТ":   ("Офталмология", "Ophthalmology"),
    "УНГ":   ("Ушно-носно-гърлени болести", "ENT / Otolaryngology"),
    "ПСИ":   ("Психиатрия", "Psychiatry"),
    "ФТ":    ("Физиотерапия", "Physiotherapy"),
    "РЕХ":   ("Рехабилитация", "Rehabilitation"),
    "ПАЛК":  ("Палиативни грижи", "Palliative care"),

    # -------------------------------------------------------------------
    # Miscellaneous clinical terms
    # -------------------------------------------------------------------
    "ФА":    ("Физическа активност", "Physical activity"),
    "ДФ":    ("Дихателна физиотерапия", "Respiratory physiotherapy"),
    "ТЛ":    ("Тромболиза", "Thrombolysis"),
    "АТ":    ("Артериална тромбоза", "Arterial thrombosis"),
    "ВТ":    ("Венозна тромбоза", "Venous thrombosis"),
    "ОА":    ("Оклузивна артериопатия", "Occlusive arteriopathy"),
    "ПАБ":   ("Периферна артериална болест", "Peripheral arterial disease"),
    "ЗА":    ("Захарен апартамент (хипогликемия)", "Hypoglycaemia"),    # informal
    "ДКА":   ("Диабетна кетоацидоза", "Diabetic ketoacidosis"),
    "ХГС":   ("Хиперосмоларно хипергликемичен статус", "Hyperosmolar hyperglycaemic state"),
    "ТГ":    ("Тиреоглобулин", "Thyroglobulin"),                        # context-dependent
    "ЩЖ":    ("Щитовидна жлеза", "Thyroid gland"),
    "ПЩА":   ("Паращитовидни аденоми", "Parathyroid adenoma"),
    "НБ":    ("Надбъбречна болест", "Adrenal disease"),
    "НК":    ("Надбъбречна кора", "Adrenal cortex"),
    "АКТХ":  ("Адренокортикотропен хормон", "ACTH"),
    "ACTH":  ("Адренокортикотропен хормон", "Adrenocorticotropic hormone"),
    "МРА":   ("Минералокортикоиден рецепторен антагонист", "Mineralocorticoid receptor antagonist"),
    "НСЕ":   ("Неврон-специфична енолаза", "Neuron-specific enolase"),
    "ПЕТ":   ("Позитронно-емисионна томография", "Positron emission tomography"),
    "ERCP":  ("Ендоскопска ретроградна холангиопанкреатография", "ERCP"),
    "ФГДС":  ("Фиброгастродуоденоскопия", "Upper GI endoscopy"),
    "ФБС":   ("Фибробронхоскопия", "Bronchoscopy"),
    "КС":    ("Кортикостероид", "Corticosteroid"),                       # context-dependent
    "ВН":    ("Венозна недостатъчност", "Venous insufficiency"),
    "ХВН":   ("Хронична венозна недостатъчност", "Chronic venous insufficiency"),
    "ВРВ":   ("Варикозни разширения на вените", "Varicose veins"),
    "ПТСС":  ("Посттромботичен синдром", "Post-thrombotic syndrome"),
    "АФС":   ("Антифосфолипиден синдром", "Antiphospholipid syndrome"),
    "ДИВ":   ("Дисеминирана интраваскуларна коагулация", "Disseminated intravascular coagulation"),
    "ДВС":   ("Дисеминирана вътресъдова коагулация", "Disseminated intravascular coagulation"),
    "ОДН":   ("Остра дихателна недостатъчност", "Acute respiratory failure"),
    "ARDS":  ("Синдром на остра дихателна недостатъчност", "Acute respiratory distress syndrome"),
    "SIRS":  ("Синдром на системна възпалителна реакция", "Systemic inflammatory response syndrome"),
    "СЕПТ":  ("Сепсис", "Sepsis"),
    "МОН":   ("Мултиорганна недостатъчност", "Multi-organ failure"),
    "ОИМ":   ("Остър исхемичен мозъчен удар", "Acute ischaemic stroke"),
    "ШОК":   ("Шок", "Shock"),

}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def expand(abbreviation: str) -> tuple[str, str] | None:
    """
    Return (full_bulgarian_term, english_translation) for the given
    abbreviation, or None if not found.  Lookup is case-sensitive for
    Latin tokens and case-insensitive for Cyrillic tokens.
    """
    result = BULGARIAN_MEDICAL_ABBREVIATIONS.get(abbreviation)
    if result is None:
        result = BULGARIAN_MEDICAL_ABBREVIATIONS.get(abbreviation.upper())
    return result


def expand_text(text: str) -> str:
    """
    Replace each known abbreviation token in *text* with
    'ABBREVIATION (full_bulgarian_term)'.
    Tokens are split on whitespace and punctuation boundaries.
    """
    import re
    token_re = re.compile(r'\b[\w\-]+\b')

    def replace(m: re.Match) -> str:
        token = m.group()
        found = expand(token)
        if found:
            return f"{token} ({found[0]})"
        return token

    return token_re.sub(replace, text)


def all_abbreviations() -> list[str]:
    """Return a sorted list of all abbreviation keys."""
    return sorted(BULGARIAN_MEDICAL_ABBREVIATIONS.keys())


def search_by_term(keyword: str) -> dict[str, tuple[str, str]]:
    """
    Search for abbreviations whose Bulgarian or English expansion contains
    *keyword* (case-insensitive).
    """
    kw = keyword.lower()
    return {
        abbr: vals
        for abbr, vals in BULGARIAN_MEDICAL_ABBREVIATIONS.items()
        if kw in vals[0].lower() or kw in vals[1].lower()
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Total abbreviations: {len(BULGARIAN_MEDICAL_ABBREVIATIONS)}\n")
    print("Sample entries:")
    samples = ["АХ", "ЕКГ", "ИМИ", "ЗД", "НСПВС", "УМБАЛ", "BMI"]
    for s in samples:
        r = expand(s)
        if r:
            print(f"  {s:10s}  →  {r[0]}  ({r[1]})")
        else:
            print(f"  {s:10s}  →  not found")

    print("\nSearch for 'инфаркт':")
    for abbr, vals in search_by_term("инфаркт").items():
        print(f"  {abbr:10s}  →  {vals[0]}")

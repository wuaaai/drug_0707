"""
expert_router.py

功能：
- 接受单个病人的 visit 列表（List[List[str]]，每个子列表为一次 visit 的 CCS/CCSCM 编码字符串）
- 使用用户定义的急症/非急症字典（set）
- 将编码映射到 19 个专家（expert id 0..18）
- 输出 top_k_chronic 个慢性病专家 id 与 top_k_acute 个急症专家 id（合并、去重、补齐）

使用：
    router = ExpertRouter(...)
    selected = router.select_experts(patient_visits)
    # selected -> List[int] 如 [2, 5, 7, 12]
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from utils import read_dataset_argument
import re
import math
import numpy as np
# ============================
# 1) 急症 / 非急症集合
#    ---- 下面变量名为 acute_codes, chronic_codes （字符串形式）
# ============================
acute_codes: Set[str] = {
    # 感染性疾病 - 生命威胁    19
    '2',    # Septicemia (except in labor) 败血症
    '76',   # Meningitis (except that caused by tuberculosis or sexually transmitted disease) 脑膜炎
    '77',   # Encephalitis (except that caused by tuberculosis or sexually transmitted disease) 脑炎
    '122',  # Pneumonia (except that caused by tuberculosis or sexually transmitted disease) 肺炎
    '129',  # Aspiration pneumonitis; food/vomitus 吸入性肺炎
    '153',  # Gastrointestinal hemorrhage 胃肠道出血
    '157',  # Acute and unspecified renal failure 急性肾衰竭
    '159',  # Urinary tract infections 尿路感染（严重时）
    '177',  # Spontaneous abortion 自然流产
    '178',  # Induced abortion 人工流产
    '179',  # Postabortion complications 流产并发症
    '180',  # Ectopic pregnancy 异位妊娠
    '182',  # Hemorrhage during pregnancy; abruptio placenta; placenta previa 妊娠出血
    '184',  # Early or threatened labor 早产或先兆临产
    '197',  # Skin and subcutaneous tissue infections 皮肤软组织感染
    '201',  # Infective arthritis and osteomyelitis 感染性关节炎和骨髓炎
    '241',  # Poisoning by psychotropic agents 精神药物中毒
    '242',  # Poisoning by other medications and drugs 其他药物中毒
    '243',  # Poisoning by nonmedicinal substances 非药物物质中毒
    
    # 心血管系统急症    9
    '100',  # Acute myocardial infarction 急性心肌梗死
    '107',  # Cardiac arrest and ventricular fibrillation 心脏骤停
    '108',  # Congestive heart failure; nonhypertensive 充血性心力衰竭
    '109',  # Acute cerebrovascular disease 急性脑血管病
    '116',  # Aortic and peripheral arterial embolism or thrombosis 动脉栓塞
    '118',  # Phlebitis; thrombophlebitis and thromboembolism 静脉炎和血栓
    '131',  # Respiratory failure; insufficiency; arrest (adult) 呼吸衰竭
    '148',  # Peritonitis and intestinal abscess 腹膜炎
    '249',  # Shock 休克
    
    # 神经系统急症   3
    '85',   # Coma; stupor; and brain damage 昏迷和脑损伤
    '227',  # Spinal cord injury 脊髓损伤
    '233',  # Intracranial injury 颅内损伤
    '83',   # Epilepsy; convulsions 癫痫
    
    # 创伤和外科急症     10
    '225',  # Joint disorders and dislocations; trauma-related 创伤性关节脱位
    '226',  # Fracture of neck of femur (hip) 股骨颈骨折
    '228',  # Skull and face fractures 颅面骨折
    '229',  # Fracture of upper limb 上肢骨折
    '230',  # Fracture of lower limb 下肢骨折
    '231',  # Other fractures 其他骨折
    '234',  # Crushing injury or internal injury 挤压伤或内伤
    '235',  # Open wounds of head; neck; and trunk 头颈躯干开放伤
    '236',  # Open wounds of extremities 肢体开放伤
    '240',  # Burns 烧伤
    '2601', # E Codes: Cut/pierce 切割刺伤
    '2602', # E Codes: Drowning/submersion 溺水
    '2603', # E Codes: Fall 跌倒
    '2604', # E Codes: Fire/burn 火灾烧伤
    '2605', # E Codes: Firearm 枪伤
    '2606', # E Codes: Machinery 机械伤
    '2607', # E Codes: Motor vehicle traffic (MVT) 机动车交通伤
    '2608', # E Codes: Pedal cyclist; not MVT 非机动车交通伤
    '2609', # E Codes: Pedestrian; not MVT 行人交通伤
    '2610', # E Codes: Transport; not MVT 其他交通伤
    '2611', # E Codes: Natural/environment 自然环境伤
    '2612', # E Codes: Overexertion 过度劳累
    '2613', # E Codes: Poisoning 中毒
    '2614', # E Codes: Struck by; against 被击中撞击
    
    # 其他急症   6
    '60',   # Acute posthemorrhagic anemia 急性失血性贫血
    '62',   # Coagulation and hemorrhagic disorders 凝血和出血性疾病
    '160',  # Calculus of urinary tract 泌尿系结石（引起梗阻时）
    '190',  # Fetal distress and abnormal forces of labor 胎儿窘迫
    '245',  # Syncope 晕厥
    '248',  # Gangrene 坏疽
    '220',  # Intrauterine hypoxia and birth asphyxia 宫内缺氧和出生窒息
    '223',  # Birth trauma 出生外伤
    '250',  # Nausea and vomiting 恶心呕吐
    '251',  # Abdominal pain 腹痛
    '252',  # Malaise and fatigue 不适和疲劳
    '253',  # Allergic reactions 过敏反应
}

# 这里把“非急症/慢性病”集合放进来（示例摘录）
chronic_codes: Set[str] = {
    '.Z',   # Overall 总体
    '.',    # No diagnosis 无诊断
    '.A',   # Invalid diagnosis 无效诊断
    '1',    # Tuberculosis 结核病（慢性）
    '3',    # Bacterial infection; unspecified site 细菌感染（非特指）
    '4',    # Mycoses 真菌病（慢性）
    '5',    # HIV infection HIV感染（慢性）
    '6',    # Hepatitis 肝炎（慢性）
    '7',    # Viral infection 病毒感染（非急性）
    '8',    # Other infections; including parasitic 其他感染（慢性）
    '9',    # Sexually transmitted infections 性传播疾病（慢性）
    '10',   # Immunizations and screening 免疫接种和筛查
    '11',   # Cancer of head and neck 头颈部癌症
    '12',   # Cancer of esophagus 食管癌
    '13',   # Cancer of stomach 胃癌
    '14',   # Cancer of colon 结肠癌
    '15',   # Cancer of rectum and anus 直肠和肛门癌
    '16',   # Cancer of liver and intrahepatic bile duct 肝癌
    '17',   # Cancer of pancreas 胰腺癌
    '18',   # Cancer of other GI organs 其他消化道癌症
    '19',   # Cancer of bronchus; lung 肺癌
    '20',   # Cancer; other respiratory 其他呼吸系统癌症
    '21',   # Cancer of bone and connective tissue 骨和结缔组织癌
    '22',   # Melanomas of skin 皮肤黑色素瘤
    '23',   # Other non-epithelial cancer of skin 其他皮肤癌
    '24',   # Cancer of breast 乳腺癌
    '25',   # Cancer of uterus 子宫癌
    '26',   # Cancer of cervix 宫颈癌
    '27',   # Cancer of ovary 卵巢癌
    '28',   # Cancer of other female genital organs 其他女性生殖器官癌
    '29',   # Cancer of prostate 前列腺癌
    '30',   # Cancer of testis 睾丸癌
    '31',   # Cancer of other male genital organs 其他男性生殖器官癌
    '32',   # Cancer of bladder 膀胱癌
    '33',   # Cancer of kidney and renal pelvis 肾和肾盂癌
    '34',   # Cancer of other urinary organs 其他泌尿器官癌
    '35',   # Cancer of brain and nervous system 脑和神经系统癌
    '36',   # Cancer of thyroid 甲状腺癌
    '37',   # Hodgkin`s disease 霍奇金病
    '38',   # Non-Hodgkin`s lymphoma 非霍奇金淋巴瘤
    '39',   # Leukemias 白血病
    '40',   # Multiple myeloma 多发性骨髓瘤
    '41',   # Cancer; other and unspecified primary 其他和未指明的原发癌
    '42',   # Secondary malignancies 继发性恶性肿瘤
    '43',   # Malignant neoplasm without specification of site 未指明部位的恶性肿瘤
    '44',   # Neoplasms of unspecified nature or uncertain behavior 性质未定或行为不确定的肿瘤
    '45',   # Maintenance chemotherapy; radiotherapy 化疗放疗维持治疗
    '46',   # Benign neoplasm of uterus 子宫良性肿瘤
    '47',   # Other and unspecified benign neoplasm 其他和未指明的良性肿瘤
    '48',   # Thyroid disorders 甲状腺疾病
    '49',   # Diabetes mellitus without complication 无并发症糖尿病
    '50',   # Diabetes mellitus with complications 有并发症糖尿病
    '51',   # Other endocrine disorders 其他内分泌疾病
    '52',   # Nutritional deficiencies 营养缺乏
    '53',   # Disorders of lipid metabolism 脂质代谢紊乱
    '54',   # Gout and other crystal arthropathies 痛风和其他晶体性关节病
    '55',   # Fluid and electrolyte disorders 水电解质紊乱
    '56',   # Cystic fibrosis 囊性纤维化
    '57',   # Immunity disorders 免疫疾病
    '58',   # Other nutritional; endocrine; and metabolic disorders 其他营养内分泌代谢疾病
    '59',   # Deficiency and other anemia 贫血
    '61',   # Sickle cell anemia 镰状细胞贫血
    '63',   # Diseases of white blood cells 白细胞疾病
    '64',   # Other hematologic conditions 其他血液病
    '650',  # Adjustment disorders 适应障碍
    '651',  # Anxiety disorders 焦虑障碍
    '652',  # Attention-deficit, conduct, and disruptive behavior disorders 注意力缺陷等行为障碍
    '653',  # Delirium, dementia, and amnestic and other cognitive disorders 谵妄痴呆等认知障碍
    '654',  # Developmental disorders 发育障碍
    '655',  # Disorders usually diagnosed in infancy, childhood, or adolescence 婴幼儿青少年期疾病
    '656',  # Impulse control disorders, NEC 冲动控制障碍
    '657',  # Mood disorders 情绪障碍
    '658',  # Personality disorders 人格障碍
    '659',  # Schizophrenia and other psychotic disorders 精神分裂症等精神病性障碍
    '660',  # Alcohol-related disorders 酒精相关障碍
    '661',  # Substance-related disorders 物质相关障碍
    '662',  # Suicide and intentional self-inflicted injury 自杀和故意自伤
    '663',  # Screening and history of mental health and substance abuse codes 精神健康和物质滥用筛查史
    '670',  # Miscellaneous mental health disorders 其他精神健康障碍
    '78',   # Other CNS infection and poliomyelitis 其他中枢神经系统感染
    '79',   # Parkinson`s disease 帕金森病
    '80',   # Multiple sclerosis 多发性硬化症
    '81',   # Other hereditary and degenerative nervous system conditions 其他遗传性和退行性神经系统疾病
    '82',   # Paralysis 瘫痪

    '84',   # Headache; including migraine 头痛
    '86',   # Cataract 白内障
    '87',   # Retinal detachments; defects; vascular occlusion; and retinopathy 视网膜疾病
    '88',   # Glaucoma 青光眼
    '89',   # Blindness and vision defects 视力缺陷
    '90',   # Inflammation; infection of eye 眼部炎症感染
    '91',   # Other eye disorders 其他眼病
    '92',   # Otitis media and related conditions 中耳炎及相关疾病
    '93',   # Conditions associated with dizziness or vertigo 眩晕相关疾病
    '94',   # Other ear and sense organ disorders 其他耳和感觉器官疾病
    '95',   # Other nervous system disorders 其他神经系统疾病
    '96',   # Heart valve disorders 心脏瓣膜疾病
    '97',   # Peri-; endo-; and myocarditis; cardiomyopathy 心肌炎心肌病
    '98',   # Essential hypertension 原发性高血压
    '99',   # Hypertension with complications and secondary hypertension 有并发症和继发性高血压
    '101',  # Coronary atherosclerosis and other heart disease 冠状动脉粥样硬化和其他心脏病
    '102',  # Nonspecific chest pain 非特异性胸痛
    '103',  # Pulmonary heart disease 肺心病
    '104',  # Other and ill-defined heart disease 其他和未指明的心脏病
    '105',  # Conduction disorders 传导障碍
    '106',  # Cardiac dysrhythmias 心律失常
    '110',  # Occlusion or stenosis of precerebral arteries 颅前动脉闭塞或狭窄
    '111',  # Other and ill-defined cerebrovascular disease 其他和未指明的脑血管病
    '112',  # Transient cerebral ischemia 短暂性脑缺血
    '113',  # Late effects of cerebrovascular disease 脑血管病后遗症
    '114',  # Peripheral and visceral atherosclerosis 外周和内脏动脉粥样硬化
    '115',  # Aortic; peripheral; and visceral artery aneurysms 动脉瘤
    '117',  # Other circulatory disease 其他循环系统疾病
    '119',  # Varicose veins of lower extremity 下肢静脉曲张
    '120',  # Hemorrhoids 痔疮
    '121',  # Other diseases of veins and lymphatics 其他静脉和淋巴管疾病
    '123',  # Influenza 流感（非重症）
    '124',  # Acute and chronic tonsillitis 扁桃体炎
    '125',  # Acute bronchitis 急性支气管炎
    '126',  # Other upper respiratory infections 其他上呼吸道感染
    '127',  # Chronic obstructive pulmonary disease and bronchiectasis 慢性阻塞性肺病
    '128',  # Asthma 哮喘
    '130',  # Pleurisy; pneumothorax; pulmonary collapse 胸膜炎气胸肺不张
    '132',  # Lung disease due to external agents 外因性肺病
    '133',  # Other lower respiratory disease 其他下呼吸道疾病
    '134',  # Other upper respiratory disease 其他上呼吸道疾病
    '135',  # Intestinal infection 肠道感染（非严重）
    '136',  # Disorders of teeth and jaw 牙齿和颌骨疾病
    '137',  # Diseases of mouth; excluding dental 口腔疾病
    '138',  # Esophageal disorders 食管疾病
    '139',  # Gastroduodenal ulcer (except hemorrhage) 胃十二指肠溃疡
    '140',  # Gastritis and duodenitis 胃炎和十二指肠炎
    '141',  # Other disorders of stomach and duodenum 其他胃十二指肠疾病
    '142',  # Appendicitis and other appendiceal conditions 阑尾炎
    '143',  # Abdominal hernia 腹疝
    '144',  # Regional enteritis and ulcerative colitis 区域性肠炎和溃疡性结肠炎
    '145',  # Intestinal obstruction without hernia 肠梗阻
    '146',  # Diverticulosis and diverticulitis 憩室病
    '147',  # Anal and rectal conditions 肛门直肠疾病
    '149',  # Biliary tract disease 胆道疾病
    '150',  # Liver disease; alcohol-related 酒精性肝病
    '151',  # Other liver diseases 其他肝病
    '152',  # Pancreatic disorders (not diabetes) 胰腺疾病（非糖尿病）
    '154',  # Noninfectious gastroenteritis 非感染性胃肠炎
    '155',  # Other gastrointestinal disorders 其他消化系统疾病
    '156',  # Nephritis; nephrosis; renal sclerosis 肾炎肾病肾硬化
    '158',  # Chronic kidney disease 慢性肾病
    '161',  # Other diseases of kidney and ureters 其他肾脏输尿管疾病
    '162',  # Other diseases of bladder and urethra 其他膀胱尿道疾病
    '163',  # Genitourinary symptoms and ill-defined conditions 泌尿生殖系统症状
    '164',  # Hyperplasia of prostate 前列腺增生
    '165',  # Inflammatory conditions of male genital organs 男性生殖器官炎症
    '166',  # Other male genital disorders 其他男性生殖器官疾病
    '167',  # Nonmalignant breast conditions 良性乳腺疾病
    '168',  # Inflammatory diseases of female pelvic organs 女性盆腔器官炎症
    '169',  # Endometriosis 子宫内膜异位症
    '170',  # Prolapse of female genital organs 女性生殖器官脱垂
    '171',  # Menstrual disorders 月经失调
    '172',  # Ovarian cyst 卵巢囊肿
    '173',  # Menopausal disorders 更年期综合征
    '174',  # Female infertility 女性不孕
    '175',  # Other female genital disorders 其他女性生殖器官疾病
    '176',  # Contraceptive and procreative management 避孕和生育管理
    '181',  # Other complications of pregnancy 妊娠其他并发症
    '183',  # Hypertension complicating pregnancy 妊娠高血压
    '185',  # Prolonged pregnancy 过期妊娠
    '186',  # Diabetes complicating pregnancy 妊娠糖尿病
    '187',  # Malposition; malpresentation 胎位异常
    '188',  # Fetopelvic disproportion; obstruction 胎儿骨盆不对称
    '189',  # Previous C-section 剖宫产史
    '191',  # Polyhydramnios and other problems of amniotic cavity 羊水过多
    '192',  # Umbilical cord complication 脐带并发症
    '193',  # OB-related trauma to perineum and vulva 产科外伤
    '194',  # Forceps delivery 产钳助产
    '195',  # Other complications of birth; puerperium 产褥期并发症
    '196',  # Other pregnancy and delivery including normal 正常妊娠分娩
    '198',  # Other inflammatory condition of skin 其他皮肤炎症
    '199',  # Chronic ulcer of skin 慢性皮肤溃疡
    '200',  # Other skin disorders 其他皮肤疾病
    '202',  # Rheumatoid arthritis and related disease 类风湿关节炎
    '203',  # Osteoarthritis 骨关节炎
    '204',  # Other non-traumatic joint disorders 其他非创伤性关节疾病
    '205',  # Spondylosis; intervertebral disc disorders 脊柱病
    '206',  # Osteoporosis 骨质疏松
    '207',  # Pathological fracture 病理性骨折
    '208',  # Acquired foot deformities 获得性足畸形
    '209',  # Other acquired deformities 其他获得性畸形
    '210',  # Systemic lupus erythematosus and connective tissue disorders 系统性红斑狼疮
    '211',  # Other connective tissue disease 其他结缔组织病
    '212',  # Other bone disease and musculoskeletal deformities 其他骨病和肌肉骨骼畸形
    '213',  # Cardiac and circulatory congenital anomalies 先天性心血管畸形
    '214',  # Digestive congenital anomalies 先天性消化道畸形
    '215',  # Genitourinary congenital anomalies 先天性泌尿生殖系统畸形
    '216',  # Nervous system congenital anomalies 先天性神经系统畸形
    '217',  # Other congenital anomalies 其他先天畸形
    '218',  # Liveborn 活产
    '219',  # Short gestation; low birth weight 早产低体重
    '221',  # Respiratory distress syndrome 呼吸窘迫综合征
    '222',  # Hemolytic jaundice and perinatal jaundice 溶血性黄疸
    '224',  # Other perinatal conditions 其他围产期疾病
    '232',  # Sprains and strains 扭伤拉伤
    '237',  # Complication of device; implant or graft 装置植入物并发症
    '238',  # Complications of surgical procedures or medical care 医疗操作并发症
    '239',  # Superficial injury; contusion 浅表损伤挫伤
    '244',  # Other injuries and conditions due to external causes 其他外因疾病
    '246',  # Fever of unknown origin 不明原因发热
    '247',  # Lymphadenitis 淋巴结炎
    '254',  # Rehabilitation care; fitting of prostheses 康复护理
    '255',  # Administrative/social admission 行政/社会入院
    '256',  # Medical examination/evaluation 医学检查评估
    '257',  # Other aftercare 其他术后护理
    '258',  # Other screening for suspected conditions 其他疑似疾病筛查
    '259',  # Residual codes; unclassified 残余代码未分类
    '260',  # E Codes: All (external causes of injury and poisoning) 外因代码

    '2615', # E Codes: Suffocation 窒息
    '2616', # E Codes: Adverse effects of medical care 医疗不良反应
    '2617', # E Codes: Adverse effects of medical drugs 药物不良反应
    '2618', # E Codes: Other specified and classifiable 其他可分类外因
    '2619', # E Codes: Other specified; NEC 其他特指外因
    '2620', # E Codes: Unspecified 未指明外因
    '2621'  # E Codes: Place of occurrence 发生地点
}

# ============================
# 2) 映射函数：CCS/CCSCM code -> expert_id (0..18)
#    - 优先使用 explicit_map（如果你有人工构建的映射表，把它放在这里）
#    - 否则使用启发式规则：按 ICD-9 一级分类的数值区间映射到 19 个桶（0..18）
#    注：该规则是可替换的；若你有权威映射表，请替换 explicit_map。
# ============================

# （示例）explicit_map: 可以把某些 code 明确映射到特定专家 id；默认为空
explicit_map: Dict[str, int] = {
    # '2': 0,    # 例如把 code '2' 固定映射到 expert 0（如果你希望）
    # '100': 6,
    # 填写你确认的映射以覆盖默认规则
}
# CCSCM 编码到 ICD-9 一级分类的完整映射字典
# 根据 ICD9CM_to_CCSCM.csv 构建
CCSCM_TO_ICD9_CHAPTER_MAP = {
    '1': '传染病和寄生虫疾病',
    '2': '传染病和寄生虫疾病',
    '3': '传染病和寄生虫疾病',
    '4': '传染病和寄生虫疾病',
    '5': '内分泌、营养、新陈代谢及免疫系统疾病',
    '6': '消化系统疾病',
    '7': '传染病和寄生虫疾病',
    '8': '传染病和寄生虫疾病',
    '9': '传染病和寄生虫疾病',
    '10': '传染病和寄生虫疾病',
    '11': '赘生物',
    '12': '赘生物',
    '13': '赘生物',
    '14': '赘生物',
    '15': '赘生物',
    '16': '赘生物',
    '17': '赘生物',
    '18': '赘生物',
    '19': '赘生物',
    '20': '赘生物',
    '21': '赘生物',
    '22': '赘生物',
    '23': '赘生物',
    '24': '赘生物',
    '25': '赘生物',
    '26': '赘生物',
    '27': '赘生物',
    '28': '赘生物',
    '29': '赘生物',
    '30': '赘生物',
    '31': '赘生物',
    '32': '赘生物',
    '33': '赘生物',
    '34': '赘生物',
    '35': '赘生物',
    '36': '赘生物',
    '37': '赘生物',
    '38': '赘生物',
    '39': '赘生物',
    '40': '赘生物',
    '41': '赘生物',
    '42': '呼吸系统疾病',
    '43': '赘生物',
    '44': '赘生物',
    '45': '赘生物',
    '46': '赘生物',
    '47': '赘生物',
    '48': '内分泌、营养、新陈代谢及免疫系统疾病',
    '49': '内分泌、营养、新陈代谢及免疫系统疾病',
    '50': '内分泌、营养、新陈代谢及免疫系统疾病',
    '51': '内分泌、营养、新陈代谢及免疫系统疾病',
    '52': '内分泌、营养、新陈代谢及免疫系统疾病',
    '53': '内分泌、营养、新陈代谢及免疫系统疾病',
    '54': '肌肉骨骼系统及结缔组织疾病',
    '55': '内分泌、营养、新陈代谢及免疫系统疾病',
    '56': '内分泌、营养、新陈代谢及免疫系统疾病',
    '57': '内分泌、营养、新陈代谢及免疫系统疾病',
    '58': '内分泌、营养、新陈代谢及免疫系统疾病',
    '59': '血液及造血器官疾病',
    '60': '血液及造血器官疾病',
    '61': '血液及造血器官疾病',
    '62': '血液及造血器官疾病',
    '63': '血液及造血器官疾病',
    '64': '血液及造血器官疾病',
    '65': '精神失常',
    '66': '精神失常',
    '67': '精神失常',
    '68': '精神失常',
    '69': '精神失常',
    '70': '精神失常',
    '71': '精神失常',
    '72': '精神失常',
    '73': '精神失常',
    '74': '精神失常',
    '75': '精神失常',
    '76': '神经系统疾病',
    '77': '神经系统疾病',
    '78': '神经系统疾病',
    '79': '神经系统疾病',
    '80': '神经系统疾病',
    '81': '神经系统疾病',
    '82': '神经系统疾病',
    '83': '神经系统疾病',
    '84': '神经系统疾病',
    '85': '神经系统疾病',
    '86': '感觉器官疾病',
    '87': '感觉器官疾病',
    '88': '感觉器官疾病',
    '89': '感觉器官疾病',
    '90': '感觉器官疾病',
    '91': '感觉器官疾病',
    '92': '感觉器官疾病',
    '93': '感觉器官疾病',
    '94': '感觉器官疾病',
    '95': '神经系统疾病',
    '96': '循环系统疾病',
    '97': '循环系统疾病',
    '98': '循环系统疾病',
    '99': '循环系统疾病',
    '100': '循环系统疾病',
    '101': '循环系统疾病',
    '102': '循环系统疾病',
    '103': '循环系统疾病',
    '104': '循环系统疾病',
    '105': '循环系统疾病',
    '106': '循环系统疾病',
    '107': '循环系统疾病',
    '108': '循环系统疾病',
    '109': '循环系统疾病',
    '110': '循环系统疾病',
    '111': '循环系统疾病',
    '112': '循环系统疾病',
    '113': '循环系统疾病',
    '114': '循环系统疾病',
    '115': '循环系统疾病',
    '116': '循环系统疾病',
    '117': '循环系统疾病',
    '118': '循环系统疾病',
    '119': '循环系统疾病',
    '120': '循环系统疾病',
    '121': '循环系统疾病',
    '122': '呼吸系统疾病',
    '123': '呼吸系统疾病',
    '124': '呼吸系统疾病',
    '125': '呼吸系统疾病',
    '126': '呼吸系统疾病',
    '127': '呼吸系统疾病',
    '128': '呼吸系统疾病',
    '129': '呼吸系统疾病',
    '130': '呼吸系统疾病',
    '131': '呼吸系统疾病',
    '132': '呼吸系统疾病',
    '133': '呼吸系统疾病',
    '134': '呼吸系统疾病',
    '135': '消化系统疾病',
    '136': '消化系统疾病',
    '137': '消化系统疾病',
    '138': '消化系统疾病',
    '139': '消化系统疾病',
    '140': '消化系统疾病',
    '141': '消化系统疾病',
    '142': '消化系统疾病',
    '143': '消化系统疾病',
    '144': '消化系统疾病',
    '145': '消化系统疾病',
    '146': '消化系统疾病',
    '147': '消化系统疾病',
    '148': '消化系统疾病',
    '149': '消化系统疾病',
    '150': '消化系统疾病',
    '151': '消化系统疾病',
    '152': '消化系统疾病',
    '153': '消化系统疾病',
    '154': '消化系统疾病',
    '155': '消化系统疾病',
    '156': '泌尿生殖系统疾病',
    '157': '泌尿生殖系统疾病',
    '158': '泌尿生殖系统疾病',
    '159': '泌尿生殖系统疾病',
    '160': '泌尿生殖系统疾病',
    '161': '泌尿生殖系统疾病',
    '162': '泌尿生殖系统疾病',
    '163': '泌尿生殖系统疾病',
    '164': '泌尿生殖系统疾病',
    '165': '泌尿生殖系统疾病',
    '166': '泌尿生殖系统疾病',
    '167': '泌尿生殖系统疾病',
    '168': '泌尿生殖系统疾病',
    '169': '泌尿生殖系统疾病',
    '170': '泌尿生殖系统疾病',
    '171': '泌尿生殖系统疾病',
    '172': '泌尿生殖系统疾病',
    '173': '泌尿生殖系统疾病',
    '174': '泌尿生殖系统疾病',
    '175': '泌尿生殖系统疾病',
    '176': '泌尿生殖系统疾病',
    '177': '妊娠、分娩和产后合并症',
    '178': '妊娠、分娩和产后合并症',
    '179': '妊娠、分娩和产后合并症',
    '180': '妊娠、分娩和产后合并症',
    '181': '妊娠、分娩和产后合并症',
    '182': '妊娠、分娩和产后合并症',
    '183': '妊娠、分娩和产后合并症',
    '184': '妊娠、分娩和产后合并症',
    '185': '妊娠、分娩和产后合并症',
    '186': '妊娠、分娩和产后合并症',
    '187': '妊娠、分娩和产后合并症',
    '188': '妊娠、分娩和产后合并症',
    '189': '妊娠、分娩和产后合并症',
    '190': '妊娠、分娩和产后合并症',
    '191': '妊娠、分娩和产后合并症',
    '192': '妊娠、分娩和产后合并症',
    '193': '妊娠、分娩和产后合并症',
    '194': '妊娠、分娩和产后合并症',
    '195': '妊娠、分娩和产后合并症',
    '196': '妊娠、分娩和产后合并症',
    '197': '皮肤及皮下组织疾病',
    '198': '皮肤及皮下组织疾病',
    '199': '皮肤及皮下组织疾病',
    '200': '皮肤及皮下组织疾病',
    '201': '皮肤及皮下组织疾病',
    '202': '肌肉骨骼系统及结缔组织疾病',
    '203': '肌肉骨骼系统及结缔组织疾病',
    '204': '肌肉骨骼系统及结缔组织疾病',
    '205': '肌肉骨骼系统及结缔组织疾病',
    '206': '肌肉骨骼系统及结缔组织疾病',
    '207': '肌肉骨骼系统及结缔组织疾病',
    '208': '肌肉骨骼系统及结缔组织疾病',
    '209': '肌肉骨骼系统及结缔组织疾病',
    '210': '肌肉骨骼系统及结缔组织疾病',
    '211': '肌肉骨骼系统及结缔组织疾病',
    '212': '肌肉骨骼系统及结缔组织疾病',
    '213': '先天性异常',
    '214': '先天性异常',
    '215': '先天性异常',
    '216': '先天性异常',
    '217': '先天性异常',
    '218': '围产期引起的某些情况',
    '219': '围产期引起的某些情况',
    '220': '围产期引起的某些情况',
    '221': '围产期引起的某些情况',
    '222': '围产期引起的某些情况',
    '223': '围产期引起的某些情况',
    '224': '传染病和寄生虫疾病',
    '225': '肌肉骨骼系统及结缔组织疾病',
    '226': '症状、体征和诊断不明的情况',
    '227': '症状、体征和诊断不明的情况',
    '228': '症状、体征和诊断不明的情况',
    '229': '症状、体征和诊断不明的情况',
    '230': '症状、体征和诊断不明的情况',
    '231': '症状、体征和诊断不明的情况',
    '232': '症状、体征和诊断不明的情况',
    '233': '症状、体征和诊断不明的情况',
    '234': '症状、体征和诊断不明的情况',
    '235': '症状、体征和诊断不明的情况',
    '236': '症状、体征和诊断不明的情况',
    '237': '症状、体征和诊断不明的情况',
    '238': '内分泌、营养、新陈代谢及免疫系统疾病',
    '239': '症状、体征和诊断不明的情况',
    '240': '症状、体征和诊断不明的情况',
    '241': '症状、体征和诊断不明的情况',
    '242': '症状、体征和诊断不明的情况',
    '243': '症状、体征和诊断不明的情况',
    '244': '症状、体征和诊断不明的情况',
    '245': '症状、体征和诊断不明的情况',
    '246': '症状、体征和诊断不明的情况',
    '247': '血液及造血器官疾病',
    '248': '症状、体征和诊断不明的情况',
    '249': '症状、体征和诊断不明的情况',
    '250': '症状、体征和诊断不明的情况',
    '251': '症状、体征和诊断不明的情况',
    '252': '皮肤及皮下组织疾病',
    '253': '呼吸系统疾病',
    '254': '症状、体征和诊断不明的情况',
    '255': '症状、体征和诊断不明的情况',
    '256': '症状、体征和诊断不明的情况',
    '257': '症状、体征和诊断不明的情况',
    '258': '症状、体征和诊断不明的情况',
    '259': '精神失常',
    '260': '影响健康状态和与保健机构接触的因素',
    '261': '影响健康状态和与保健机构接触的因素',
    '262': '影响健康状态和与保健机构接触的因素',
    '263': '影响健康状态和与保健机构接触的因素',
    '264': '影响健康状态和与保健机构接触的因素',
    '265': '影响健康状态和与保健机构接触的因素',
    '266': '影响健康状态和与保健机构接触的因素',
    '267': '影响健康状态和与保健机构接触的因素',
    '268': '影响健康状态和与保健机构接触的因素',
    '269': '影响健康状态和与保健机构接触的因素',
    '270': '影响健康状态和与保健机构接触的因素',
    '271': '影响健康状态和与保健机构接触的因素',
    '272': '影响健康状态和与保健机构接触的因素',
    '273': '影响健康状态和与保健机构接触的因素',
    '274': '影响健康状态和与保健机构接触的因素',
    '275': '影响健康状态和与保健机构接触的因素',
    '276': '影响健康状态和与保健机构接触的因素',
    '277': '影响健康状态和与保健机构接触的因素',
    '278': '影响健康状态和与保健机构接触的因素',
    '279': '影响健康状态和与保健机构接触的因素',
    '280': '影响健康状态和与保健机构接触的因素',
    '281': '影响健康状态和与保健机构接触的因素',
    '282': '影响健康状态和与保健机构接触的因素',
    '283': '影响健康状态和与保健机构接触的因素',
    '284': '影响健康状态和与保健机构接触的因素',
    '285': '影响健康状态和与保健机构接触的因素',
    '286': '影响健康状态和与保健机构接触的因素',
    '287': '影响健康状态和与保健机构接触的因素',
    '288': '影响健康状态和与保健机构接触的因素',
    '289': '影响健康状态和与保健机构接触的因素',
    '290': '影响健康状态和与保健机构接触的因素',
    '291': '影响健康状态和与保健机构接触的因素',
    '292': '影响健康状态和与保健机构接触的因素',
    '293': '影响健康状态和与保健机构接触的因素',
    '294': '影响健康状态和与保健机构接触的因素',
    '295': '影响健康状态和与保健机构接触的因素',
    '296': '影响健康状态和与保健机构接触的因素',
    '297': '影响健康状态和与保健机构接触的因素',
    '298': '影响健康状态和与保健机构接触的因素',
    '299': '影响健康状态和与保健机构接触的因素',
    '300': '影响健康状态和与保健机构接触的因素',
    '301': '影响健康状态和与保健机构接触的因素',
    '302': '影响健康状态和与保健机构接触的因素',
    '303': '影响健康状态和与保健机构接触的因素',
    '304': '影响健康状态和与保健机构接触的因素',
    '305': '影响健康状态和与保健机构接触的因素',
    '306': '影响健康状态和与保健机构接触的因素',
    '307': '影响健康状态和与保健机构接触的因素',
    '308': '影响健康状态和与保健机构接触的因素',
    '309': '影响健康状态和与保健机构接触的因素',
    '310': '影响健康状态和与保健机构接触的因素',
    '311': '影响健康状态和与保健机构接触的因素',
    '312': '影响健康状态和与保健机构接触的因素',
    '313': '影响健康状态和与保健机构接触的因素',
    '314': '影响健康状态和与保健机构接触的因素',
    '315': '影响健康状态和与保健机构接触的因素',
    '316': '影响健康状态和与保健机构接触的因素',
    '317': '影响健康状态和与保健机构接触的因素',
    '318': '影响健康状态和与保健机构接触的因素',
    '319': '影响健康状态和与保健机构接触的因素',
    '320': '影响健康状态和与保健机构接触的因素',
    '321': '影响健康状态和与保健机构接触的因素',
    '322': '影响健康状态和与保健机构接触的因素',
    '323': '影响健康状态和与保健机构接触的因素',
    '324': '影响健康状态和与保健机构接触的因素',
    '325': '影响健康状态和与保健机构接触的因素',
    '326': '影响健康状态和与保健机构接触的因素',
    '327': '影响健康状态和与保健机构接触的因素',
    '650': '症状、体征和诊断不明的情况',
    '651': '精神失常',
    '652': '精神失常',
    '653': '精神失常',
    '654': '精神失常',
    '655': '精神失常',
    '656': '精神失常',
    '657': '精神失常',
    '658': '精神失常',
    '659': '精神失常',
    '660': '循环系统疾病',
    '661': '损伤和中毒',
    '662': '损伤和中毒的外部原因',
    '663': '神经系统疾病',
    '670': '精神失常',
    '2601': '损伤和中毒的外部原因',
    '2602': '损伤和中毒的外部原因',
    '2603': '损伤和中毒的外部原因',
    '2604': '损伤和中毒的外部原因',
    '2605': '损伤和中毒的外部原因',
    '2606': '损伤和中毒的外部原因',
    '2607': '损伤和中毒的外部原因',
    '2608': '损伤和中毒的外部原因',
    '2609': '损伤和中毒的外部原因',
    '2610': '损伤和中毒的外部原因',
    '2611': '损伤和中毒的外部原因',
    '2612': '损伤和中毒的外部原因',
    '2613': '损伤和中毒的外部原因',
    '2614': '损伤和中毒的外部原因',
    '2615': '损伤和中毒的外部原因',
    '2616': '损伤和中毒的外部原因',
    '2617': '损伤和中毒的外部原因',
    '2618': '损伤和中毒的外部原因',
    '2619': '损伤和中毒的外部原因',
    '2620': '损伤和中毒的外部原因',
    '2621': '损伤和中毒的外部原因'
}
CHAPTER_TO_EXPERT_ID_MAP_ICD9 = {
    # 这是根据您原有的 map_ccs_to_expert 函数中的注释反推出来的
    # 请您务必核对这个映射是否符合您的 19 个专家的定义！
    
    "传染病和寄生虫疾病": 0,
    "赘生物": 1,
    "内分泌、营养、新陈代谢及免疫系统疾病": 2,
    "血液及造血器官疾病": 3,
    "精神失常": 4,
    "神经系统疾病": 5, 
    "感觉器官疾病": 6,  
    "循环系统疾病": 7,
    "呼吸系统疾病": 8,
    "消化系统疾病": 9,
    "泌尿生殖系统疾病": 10,
    "妊娠、分娩和产后合并症": 11,
    "皮肤及皮下组织疾病": 12,
    "肌肉骨骼系统及结缔组织疾病": 13,
    "先天性异常": 14,
    "围产期引起的某些情况": 15,
    "症状、体征和诊断不明的情况": 16,
    "损伤和中毒": 17,
    
    # 以下是 V/E/未知 分类
    "影响健康状态和与保健机构接触的因素": 18, # V-Codes
    "损伤和中毒的外部原因": 18,               # E-Codes
    "未知分类": 18                         # Fallback
}
CCSCM_TO_ICD10_CHAPTER_MAP = {
    '1': 'I 某些传染病和寄生虫病',
    '2': 'I 某些传染病和寄生虫病',
    '3': 'I 某些传染病和寄生虫病',
    '4': 'I 某些传染病和寄生虫病',
    '5': 'IV 内分泌、营养和代谢疾病',
    '6': 'XI 消化系统疾病',
    '7': 'I 某些传染病和寄生虫病',
    '8': 'I 某些传染病和寄生虫病',
    '9': 'I 某些传染病和寄生虫病',
    '10': 'XXI 影响健康状况和接触健康服务的因素',
    '11': 'II 肿瘤',
    '12': 'II 肿瘤',
    '13': 'II 肿瘤',
    '14': 'II 肿瘤',
    '15': 'II 肿瘤',
    '16': 'II 肿瘤',
    '17': 'II 肿瘤',
    '18': 'II 肿瘤',
    '19': 'II 肿瘤',
    '20': 'II 肿瘤',
    '21': 'II 肿瘤',
    '22': 'II 肿瘤',
    '23': 'II 肿瘤',
    '24': 'II 肿瘤',
    '25': 'II 肿瘤',
    '26': 'II 肿瘤',
    '27': 'II 肿瘤',
    '28': 'II 肿瘤',
    '29': 'II 肿瘤',
    '30': 'II 肿瘤',
    '31': 'II 肿瘤',
    '32': 'II 肿瘤',
    '33': 'II 肿瘤',
    '34': 'II 肿瘤',
    '35': 'II 肿瘤',
    '36': 'II 肿瘤',
    '37': 'II 肿瘤',
    '38': 'II 肿瘤',
    '39': 'II 肿瘤',
    '40': 'II 肿瘤',
    '41': 'II 肿瘤',
    '42': 'X 呼吸系统疾病',
    '43': 'II 肿瘤',
    '44': 'II 肿瘤',
    '45': 'II 肿瘤',
    '46': 'II 肿瘤',
    '47': 'II 肿瘤',
    '48': 'IV 内分泌、营养和代谢疾病',
    '49': 'IV 内分泌、营养和代谢疾病',
    '50': 'IV 内分泌、营养和代谢疾病',
    '51': 'IV 内分泌、营养和代谢疾病',
    '52': 'IV 内分泌、营养和代谢疾病',
    '53': 'IV 内分泌、营养和代谢疾病',
    '54': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '55': 'IV 内分泌、营养和代谢疾病',
    '56': 'X 呼吸系统疾病',
    '57': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '58': 'IV 内分泌、营养和代谢疾病',
    '59': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '60': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '61': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '62': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '63': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '64': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '65': 'V 精神和行为障碍',
    '66': 'V 精神和行为障碍',
    '67': 'V 精神和行为障碍',
    '68': '未知分类 (ICD10_22)',
    '69': 'V 精神和行为障碍',
    '70': 'V 精神和行为障碍',
    '71': 'V 精神和行为障碍',
    '72': 'V 精神和行为障碍',
    '73': 'V 精神和行为障碍',
    '74': 'V 精神和行为障碍',
    '75': 'V 精神和行为障碍',
    '76': 'VI 神经系统疾病',
    '77': 'VI 神经系统疾病',
    '78': 'VI 神经系统疾病',
    '79': 'VI 神经系统疾病',
    '80': 'VI 神经系统疾病',
    '81': 'VI 神经系统疾病',
    '82': 'VI 神经系统疾病',
    '83': 'VI 神经系统疾病',
    '84': 'VI 神经系统疾病',
    '85': 'VI 神经系统疾病',
    '86': 'VII 眼和附器疾病',
    '87': 'VII 眼和附器疾病',
    '88': 'VII 眼和附器疾病',
    '89': 'VII 眼和附器疾病',
    '90': 'VII 眼和附器疾病',
    '91': 'VII 眼和附器疾病',
    '92': 'VIII 耳和乳突疾病',
    '93': 'VIII 耳和乳突疾病',
    '94': 'VIII 耳和乳突疾病',
    '95': 'VI 神经系统疾病',
    '96': 'IX 循环系统疾病',
    '97': 'IX 循环系统疾病',
    '98': 'IX 循环系统疾病',
    '99': 'IX 循环系统疾病',
    '100': 'IX 循环系统疾病',
    '101': 'IX 循环系统疾病',
    '102': 'IX 循环系统疾病',
    '103': 'IX 循环系统疾病',
    '104': 'IX 循环系统疾病',
    '105': 'IX 循环系统疾病',
    '106': 'IX 循环系统疾病',
    '107': 'IX 循环系统疾病',
    '108': 'IX 循环系统疾病',
    '109': 'IX 循环系统疾病',
    '110': 'IX 循环系统疾病',
    '111': 'IX 循环系统疾病',
    '112': 'IX 循环系统疾病',
    '113': 'IX 循环系统疾病',
    '114': 'IX 循环系统疾病',
    '115': 'IX 循环系统疾病',
    '116': 'IX 循环系统疾病',
    '117': 'IX 循环系统疾病',
    '118': 'IX 循环系统疾病',
    '119': 'IX 循环系统疾病',
    '120': 'XI 消化系统疾病',
    '121': 'IX 循环系统疾病',
    '122': 'X 呼吸系统疾病',
    '123': 'X 呼吸系统疾病',
    '124': 'X 呼吸系统疾病',
    '125': 'X 呼吸系统疾病',
    '126': 'X 呼吸系统疾病',
    '127': 'X 呼吸系统疾病',
    '128': 'X 呼吸系统疾病',
    '129': 'X 呼吸系统疾病',
    '130': 'X 呼吸系统疾病',
    '131': 'X 呼吸系统疾病',
    '132': 'X 呼吸系统疾病',
    '133': 'X 呼吸系统疾病',
    '134': 'X 呼吸系统疾病',
    '135': 'XI 消化系统疾病',
    '136': 'XI 消化系统疾病',
    '137': 'XI 消化系统疾病',
    '138': 'XI 消化系统疾病',
    '139': 'XI 消化系统疾病',
    '140': 'XI 消化系统疾病',
    '141': 'XI 消化系统疾病',
    '142': 'XI 消化系统疾病',
    '143': 'XI 消化系统疾病',
    '144': 'XI 消化系统疾病',
    '145': 'XI 消化系统疾病',
    '146': 'XI 消化系统疾病',
    '147': 'XI 消化系统疾病',
    '148': 'XI 消化系统疾病',
    '149': 'XI 消化系统疾病',
    '150': 'XI 消化系统疾病',
    '151': 'XI 消化系统疾病',
    '152': 'XI 消化系统疾病',
    '153': 'XI 消化系统疾病',
    '154': 'XI 消化系统疾病',
    '155': 'XI 消化系统疾病',
    '156': 'XIV 泌尿生殖系统疾病',
    '157': 'XIV 泌尿生殖系统疾病',
    '158': 'XIV 泌尿生殖系统疾病',
    '159': 'XIV 泌尿生殖系统疾病',
    '160': 'XIV 泌尿生殖系统疾病',
    '161': 'XIV 泌尿生殖系统疾病',
    '162': 'XIV 泌尿生殖系统疾病',
    '163': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '164': 'XIV 泌尿生殖系统疾病',
    '165': 'XIV 泌尿生殖系统疾病',
    '166': 'XIV 泌尿生殖系统疾病',
    '167': 'XIV 泌尿生殖系统疾病',
    '168': 'XIV 泌尿生殖系统疾病',
    '169': 'XIV 泌尿生殖系统疾病',
    '170': 'XIV 泌尿生殖系统疾病',
    '171': 'XIV 泌尿生殖系统疾病',
    '172': 'XIV 泌尿生殖系统疾病',
    '173': 'XIV 泌尿生殖系统疾病',
    '174': 'XIV 泌尿生殖系统疾病',
    '175': 'XIV 泌尿生殖系统疾病',
    '176': 'XXI 影响健康状况和接触健康服务的因素',
    '177': 'XV 妊娠、分娩和产褥期',
    '178': 'XXI 影响健康状况和接触健康服务的因素',
    '179': 'XV 妊娠、分娩和产褥期',
    '180': 'XV 妊娠、分娩和产褥期',
    '181': 'XV 妊娠、分娩和产褥期',
    '182': 'XV 妊娠、分娩和产褥期',
    '183': 'XV 妊娠、分娩和产褥期',
    '184': 'XV 妊娠、分娩和产褥期',
    '185': 'XV 妊娠、分娩和产褥期',
    '186': 'XV 妊娠、分娩和产褥期',
    '187': 'XV 妊娠、分娩和产褥期',
    '188': 'XV 妊娠、分娩和产褥期',
    '189': 'XV 妊娠、分娩和产褥期',
    '190': 'XV 妊娠、分娩和产褥期',
    '191': 'XV 妊娠、分娩和产褥期',
    '192': 'XV 妊娠、分娩和产褥期',
    '193': 'XV 妊娠、分娩和产褥期',
    '194': 'XXI 影响健康状况和接触健康服务的因素',
    '195': 'XV 妊娠、分娩和产褥期',
    '196': 'XXI 影响健康状况和接触健康服务的因素',
    '197': 'XII 皮肤和皮下组织疾病',
    '198': 'XII 皮肤和皮下组织疾病',
    '199': 'XII 皮肤和皮下组织疾病',
    '200': 'XII 皮肤和皮下组织疾病',
    '201': 'XII 皮肤和皮下组织疾病',
    '202': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '203': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '204': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '205': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '206': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '207': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '208': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '209': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '210': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '211': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '212': 'XIII 肌肉骨骼系统和结缔组织疾病',
    '213': 'XVII 先天性畸形、变形和染色体异常',
    '214': 'XVII 先天性畸形、变形和染色体异常',
    '215': 'XVII 先天性畸形、变形和染色体异常',
    '216': 'XVII 先天性畸形、变形和染色体异常',
    '217': 'XVII 先天性畸形、变形和染色体异常',
    '218': 'XXI 影响健康状况和接触健康服务的因素',
    '219': 'XVI 起源于围生期的某些疾病',
    '220': 'XVI 起源于围生期的某些疾病',
    '221': 'XVI 起源于围生期的某些疾病',
    '222': 'XVI 起源于围生期的某些疾病',
    '223': 'XVI 起源于围生期的某些疾病',
    '224': 'XVI 起源于围生期的某些疾病',
    '225': 'XIX 损伤、中毒和外因的某些其它结果',
    '226': 'XIX 损伤、中毒和外因的某些其它结果',
    '227': 'XIX 损伤、中毒和外因的某些其它结果',
    '228': 'XIX 损伤、中毒和外因的某些其它结果',
    '229': 'XIX 损伤、中毒和外因的某些其它结果',
    '230': 'XIX 损伤、中毒和外因的某些其它结果',
    '231': 'XIX 损伤、中毒和外因的某些其它结果',
    '232': 'XIX 损伤、中毒和外因的某些其它结果',
    '233': 'XIX 损伤、中毒和外因的某些其它结果',
    '234': 'XIX 损伤、中毒和外因的某些其它结果',
    '235': 'XIX 损伤、中毒和外因的某些其它结果',
    '236': 'XIX 损伤、中毒和外因的某些其它结果',
    '237': 'XIX 损伤、中毒和外因的某些其它结果',
    '238': 'XIX 损伤、中毒和外因的某些其它结果',
    '239': 'XIX 损伤、中毒和外因的某些其它结果',
    '240': 'XIX 损伤、中毒和外因的某些其它结果',
    '241': 'XIX 损伤、中毒和外因的某些其它结果',
    '242': 'XIX 损伤、中毒和外因的某些其它结果',
    '243': 'XIX 损伤、中毒和外因的某些其它结果',
    '244': 'XIX 损伤、中毒和外因的某些其它结果',
    '245': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '246': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '247': 'III 血液和造血器官疾病以及某些涉及免疫机能的异常',
    '248': 'IX 循环系统疾病',
    '249': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '250': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '251': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '252': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '253': 'XII 皮肤和皮下组织疾病',
    '254': 'XXI 影响健康状况和接触健康服务的因素',
    '255': 'XXI 影响健康状况和接触健康服务的因素',
    '256': 'XXI 影响健康状况和接触健康服务的因素',
    '257': 'XXI 影响健康状况和接触健康服务的因素',
    '258': 'XXI 影响健康状况和接触健康服务的因素',
    '259': 'XVIII 症状、体征和异常的临床和化验结果 NEC',
    '260': 'XXI 影响健康状况和接触健康服务的因素',
    '261': 'XXI 影响健康状况和接触健康服务的因素',
    '262': 'XXI 影响健康状况和接触健康服务的因素',
    '263': 'XXI 影响健康状况和接触健康服务的因素',
    '264': 'XXI 影响健康状况和接触健康服务的因素',
    '265': 'XXI 影响健康状况和接触健康服务的因素',
    '266': 'XXI 影响健康状况和接触健康服务的因素',
    '267': 'XXI 影响健康状况和接触健康服务的因素',
    '268': 'XXI 影响健康状况和接触健康服务的因素',
    '269': 'XXI 影响健康状况和接触健康服务的因素',
    '270': 'XXI 影响健康状况和接触健康服务的因素',
    '271': 'XXI 影响健康状况和接触健康服务的因素',
    '272': 'XXI 影响健康状况和接触健康服务的因素',
    '273': 'XXI 影响健康状况和接触健康服务的因素',
    '274': 'XXI 影响健康状况和接触健康服务的因素',
    '275': 'XXI 影响健康状况和接触健康服务的因素',
    '276': 'XXI 影响健康状况和接触健康服务的因素',
    '277': 'XXI 影响健康状况和接触健康服务的因素',
    '278': 'XXI 影响健康状况和接触健康服务的因素',
    '279': 'XXI 影响健康状况和接触健康服务的因素',
    '280': 'XXI 影响健康状况和接触健康服务的因素',
    '281': 'XXI 影响健康状况和接触健康服务的因素',
    '282': 'XXI 影响健康状况和接触健康服务的因素',
    '283': 'XXI 影响健康状况和接触健康服务的因素',
    '284': 'XXI 影响健康状况和接触健康服务的因素',
    '285': 'XXI 影响健康状况和接触健康服务的因素',
    '286': 'XXI 影响健康状况和接触健康服务的因素',
    '287': 'XXI 影响健康状况和接触健康服务的因素',
    '288': 'XXI 影响健康状况和接触健康服务的因素',
    '289': 'XXI 影响健康状况和接触健康服务的因素',
    '290': 'XXI 影响健康状况和接触健康服务的因素',
    '291': 'XXI 影响健康状况和接触健康服务的因素',
    '292': 'XXI 影响健康状况和接触健康服务的因素',
    '293': 'XXI 影响健康状况和接触健康服务的因素',
    '294': 'XXI 影响健康状况和接触健康服务的因素',
    '295': 'XXI 影响健康状况和接触健康服务的因素',
    '296': 'XXI 影响健康状况和接触健康服务的因素',
    '297': 'XXI 影响健康状况和接触健康服务的因素',
    '298': 'XXI 影响健康状况和接触健康服务的因素',
    '299': 'XXI 影响健康状况和接触健康服务的因素',
    '300': 'XXI 影响健康状况和接触健康服务的因素',
    '301': 'XXI 影响健康状况和接触健康服务的因素',
    '302': 'XXI 影响健康状况和接触健康服务的因素',
    '303': 'XXI 影响健康状况和接触健康服务的因素',
    '304': 'XXI 影响健康状况和接触健康服务的因素',
    '305': 'XXI 影响健康状况和接触健康服务的因素',
    '306': 'XXI 影响健康状况和接触健康服务的因素',
    '307': 'XXI 影响健康状况和接触健康服务的因素',
    '308': 'XXI 影响健康状况和接触健康服务的因素',
    '309': 'XXI 影响健康状况和接触健康服务的因素',
    '310': 'XXI 影响健康状况和接触健康服务的因素',
    '311': 'XXI 影响健康状况和接触健康服务的因素',
    '312': 'XXI 影响健康状况和接触健康服务的因素',
    '313': 'XXI 影响健康状况和接触健康服务的因素',
    '314': 'XXI 影响健康状况和接触健康服务的因素',
    '315': 'XXI 影响健康状况和接触健康服务的因素',
    '316': 'XXI 影响健康状况和接触健康服务的因素',
    '317': 'XXI 影响健康状况和接触健康服务的因素',
    '318': 'XXI 影响健康状况和接触健康服务的因素',
    '319': 'XXI 影响健康状况和接触健康服务的因素',
    '320': 'XXI 影响健康状况和接触健康服务的因素',
    '321': 'XXI 影响健康状况和接触健康服务的因素',
    '322': 'XXI 影响健康状况和接触健康服务的因素',
    '323': 'XXI 影响健康状况和接触健康服务的因素',
    '324': 'XXI 影响健康状况和接触健康服务的因素',
    '325': 'XXI 影响健康状况和接触健康服务的因素',
    '326': 'XXI 影响健康状况和接触健康服务的因素',
    '327': 'XXI 影响健康状况和接触健康服务的因素',
    '650': 'V 精神和行为障碍',
    '651': 'V 精神和行为障碍',
    '652': 'V 精神和行为障碍',
    '653': 'V 精神和行为障碍',
    '654': 'V 精神和行为障碍',
    '655': 'V 精神和行为障碍',
    '656': 'V 精神和行为障碍',
    '657': 'V 精神和行为障碍',
    '658': 'V 精神和行为障碍',
    '659': 'V 精神和行为障碍',
    '660': 'V 精神和行为障碍',
    '661': 'V 精神和行为障碍',
    '662': 'XIX 损伤、中毒和外因的某些其它结果',
    '663': 'XXI 影响健康状况和接触健康服务的因素',
    '670': 'V 精神和行为障碍',
    '2601': 'XX 发病和死亡的外因',
    '2602': 'XX 发病和死亡的外因',
    '2603': 'XX 发病和死亡的外因',
    '2604': 'XX 发病和死亡的外因',
    '2605': 'XX 发病和死亡的外因',
    '2606': 'XX 发病和死亡的外因',
    '2607': 'XX 发病和死亡的外因',
    '2608': 'XX 发病和死亡的外因',
    '2609': 'XX 发病和死亡的外因',
    '2610': 'XX 发病和死亡的外因',
    '2611': 'XX 发病和死亡的外因',
    '2612': 'XX 发病和死亡的外因',
    '2613': 'XIX 损伤、中毒和外因的某些其它结果',
    '2614': 'XX 发病和死亡的外因',
    '2615': 'XX 发病和死亡的外因',
    '2616': 'XXI 影响健康状况和接触健康服务的因素',
    '2617': 'XIX 损伤、中毒和外因的某些其它结果',
    '2618': 'XX 发病和死亡的外因',
    '2619': 'XX 发病和死亡的外因',
    '2620': 'XX 发病和死亡的外因',
    '2621': 'XX 发病和死亡的外因',
}
CHAPTER_TO_EXPERT_ID_MAP_ICD10 = {
    'I 某些传染病和寄生虫病': 0,
    'II 肿瘤': 1,
    'III 血液和造血器官疾病以及某些涉及免疫机能的异常': 2,
    'IV 内分泌、营养和代谢疾病': 3,
    'V 精神和行为障碍': 4,
    'VI 神经系统疾病': 5,
    'VII 眼和附器疾病': 6,
    'VIII 耳和乳突疾病': 7,
    'IX 循环系统疾病': 8,
    'X 呼吸系统疾病': 9,
    'XI 消化系统疾病': 10,
    'XII 皮肤和皮下组织疾病': 11,
    'XIII 肌肉骨骼系统和结缔组织疾病': 12,
    'XIV 泌尿生殖系统疾病': 13,
    'XV 妊娠、分娩和产褥期': 14,
    'XVI 起源于围生期的某些疾病': 15,
    'XVII 先天性畸形、变形和染色体异常': 16,
    'XVIII 症状、体征和异常的临床和化验结果 NEC': 17,
    'XIX 损伤、中毒和外因的某些其它结果': 18,
    'XX 发病和死亡的外因': 19,
    'XXI 影响健康状况和接触健康服务的因素': 20,
    'XXII 特殊用途编码': 21,
    '未知分类 (ICD10_22)': 21, # Fallback: 归入最后一个专家(21)
}
def map_ccs_to_expert(code: str) -> int:
    """
    把单个 CCS/CCSCM code 映射到 expert_id in [0, 18]（共19个专家）。
    
    新规则（优先级）：
     1) 如果 explicit_map 中存在，直接返回 explicit_map[code]
     2) 使用 CCSCM_TO_ICD9_CHAPTER_MAP 查到一级分类名称
     3) 使用 CHAPTER_TO_EXPERT_ID_MAP 查到最终的 expert_id
    """
    code_s = str(code).strip()
    
    # 1. 检查人工指定的映射 (优先级最高)
    if code_s in explicit_map:
        return explicit_map[code_s]
    current_dataset = read_dataset_argument()
    if current_dataset == 'mimic3':
        # 2a. 从 CCSCM 映射到 ICD-9 一级分类名称
        #    (如果找不到，默认为 "未知分类")
        chapter_name = CCSCM_TO_ICD9_CHAPTER_MAP.get(code_s, "未知分类")
        
        # 3a. 从分类名称映射到 Expert ID
        #    (如果找不到，默认为 18)
        expert_id = CHAPTER_TO_EXPERT_ID_MAP_ICD9.get(chapter_name, 18)
    elif current_dataset == 'mimic4':
        # --- 使用 ICD-10 (22专家) 映射 ---
        # 2b. 从 CCSCM 映射到 ICD-10 一级分类名称
        chapter_name = CCSCM_TO_ICD10_CHAPTER_MAP.get(code_s, "UNKNOWN_CATEGORY_ICD10_22") # 使用 ICD10 未知分类常量
        # 3b. 从分类名称映射到 Expert ID (0-21)
        expert_id = CHAPTER_TO_EXPERT_ID_MAP_ICD10.get(chapter_name, 21) # ICD-10 未知映射到 21
    return expert_id
# ============================
# 3) ExpertRouter: 主模块
# ============================
class ExpertRouter:
    """
    Expert selection (routing) module.

    Parameters
    ----------
    num_experts : int
        专家数量（默认 19）
    acute_codes_set : Set[str]
        急症编码集合（CCS/CCSCM code 字符串）
    chronic_codes_set : Set[str]
        非急症（慢性）编码集合
    recent_k : int
        计算急症信号时采用的最近 K 次 visit（默认 3）
    decay_lambda : float

    衰减系数（默认 0.1），控制急症专家的时间衰减强度
    """

    def __init__(
        self,
        # num_experts: int = 19,
        acute_codes_set: Optional[Set[str]] = None,
        chronic_codes_set: Optional[Set[str]] = None,
        recent_k: int = 3,
        decay_lambda: float = 0.1,   # --- 修改: 增加衰减系数
        # matrix_path: str = 'co_occurrence_matrix.npy'
    ):
        current_dataset = read_dataset_argument()
        if current_dataset == 'mimic3':
            num_experts = 19
            matrix_path = 'co_occurrence_matrix_mimic3.npy'
            print(f"ExpertRouter Info: 检测到数据集 '{current_dataset}', 设置 num_experts = {num_experts}, matrix_path = '{matrix_path}'")
        elif current_dataset == 'mimic4':
            num_experts = 22
            matrix_path = 'co_occurrence_matrix_mimic4.npy' 
            print(f"ExpertRouter Info: 检测到数据集 '{current_dataset}', 设置 num_experts = {num_experts}, matrix_path = '{matrix_path}'")
        else:
            print(f"警告 (ExpertRouter.__init__): 未知的数据集 '{current_dataset}'。将回退使用 mimic3 的设置 (19 experts)。")
            num_experts = 19 # 默认回退到 mimic3 的设置
            matrix_path = 'co_occurrence_matrix_mimic3.npy' 
        self.num_experts = num_experts
        self.acute_codes = acute_codes_set or set()
        self.chronic_codes = chronic_codes_set or set()
        self.recent_k = max(1, int(recent_k))
        self.decay_lambda = decay_lambda  # --- 修改: 保存衰减系数
        # === 修改 2: 加载关联矩阵，并从中自动推断热门专家排序 ===
        try:
            self.co_occurrence_matrix = np.load(matrix_path)
            np.fill_diagonal(self.co_occurrence_matrix, 0)
            print("专家关联矩阵加载成功。")

            # --- 核心改动：根据矩阵自动计算热门专家排序 ---
            # 计算每个专家（每行）的共现总数
            expert_popularity_scores = np.sum(self.co_occurrence_matrix, axis=1)
            # 根据分数从高到低排序，得到专家ID列表
            self.popular_experts = np.argsort(expert_popularity_scores)[::-1].tolist()
            print("已根据关联矩阵自动生成热门专家排序。")

        except FileNotFoundError:
            print(f"警告: 在路径 {matrix_path} 未找到专家关联矩阵。补齐逻辑将使用默认ID排序。")
            self.co_occurrence_matrix = np.zeros((num_experts, num_experts), dtype=int)
            # 如果文件不存在，回退到使用默认的ID顺序
            self.popular_experts = list(range(num_experts))

            
    # ---------- 辅助方法 ----------
    @staticmethod
    def _top_k_from_counter(counter: Counter, k: int, num_experts: int) -> List[int]:
        """
        从 Counter（key=expert_id, value=count）中返回 top-k expert ids（长度 = k）。
        - 按 count 降序；若 count 相同则按 expert_id 升序（确保确定性）。
        - 若 counter 中不同专家少于 k，则后续函数负责补齐。
        """
        # prepare (expert_id, count) for all experts (including zero-count)
        items = [(eid, counter.get(eid, 0)) for eid in range(num_experts)]
        # sort by (-count, eid)
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
        topk = [eid for eid, _ in items_sorted[:k]]
        return topk

    def _count_experts_from_visits(self, patient_visits: List[List[str]], which: str = 'chronic') -> Counter:
        """
        根据 patient_visits 统计每个 expert 的出现次数。
        - which == 'chronic' : 用所有 visit，且只统计在 chronic_codes 集合里的编码
        - which == 'acute'   : 用最近 self.recent_k 个 visit，且只统计在 acute_codes 集合里的编码
        返回 Counter(expert_id -> count)
        """
        assert which in ('chronic', 'acute')
        counter = Counter()
        if which == 'chronic':
            for visit in patient_visits:
                for code in visit:
                    # 仅统计在慢性集合中的编码（如果你想改为“所有编码都映射后再选择慢性专家”，可修改这里）
                    if code in self.chronic_codes:
                        eid = map_ccs_to_expert(code)
                        counter[eid] += 1
        else:
            # acute: 最近 K visits
            recent_visits = patient_visits[-self.recent_k:] if len(patient_visits) >= self.recent_k else patient_visits
            T = len(recent_visits) - 1  # 最后一次 visit 索引

            for t, visit in enumerate(recent_visits):

                # 衰减权重: 越新的 visit 权重越大

                w = math.exp(-self.decay_lambda * (T - t))
                for code in visit:
                    if code in self.acute_codes:
                        eid = map_ccs_to_expert(code)
                        counter[eid] += w
        return counter

    # ---------- 主选择方法 ----------
    def select_experts(
        self,
        patient_visits: List[List[str]],
        top_k_chronic: int = 2,
        top_k_acute: int = 2,
        deduplicate: bool = True,
        ensure_length: bool = True
    ) -> List[int]:
        """
        主方法：返回专家 id 列表。
        参数：
          - patient_visits: List[List[str]]，按时间顺序排列的 visits（最早 -> 最新）
          - top_k_chronic: 从慢性通道选多少个（默认 2）
          - top_k_acute: 从急症通道选多少个（默认 2）
          - deduplicate: 若 True，则去重（慢性优先，急症补齐）
          - ensure_length: 若 True，则保证返回长度为 top_k_chronic + top_k_acute（通过补齐）
        返回：
          - List[int]：选出的专家 id 列表（长度可能 < 固定长度，除非 ensure_length=True）
        """
        # 1) 统计
        chronic_counter = self._count_experts_from_visits(patient_visits, which='chronic')
        acute_counter = self._count_experts_from_visits(patient_visits, which='acute')

        # 2) top-k 各自选出
        chronic_topk = self._top_k_from_counter(chronic_counter, top_k_chronic, self.num_experts)
        acute_topk = self._top_k_from_counter(acute_counter, top_k_acute, self.num_experts)

        # 3) 合并并按规则处理去重/补齐
        if deduplicate:
            selected = []
            # 慢性优先
            for eid in chronic_topk:
                if eid not in selected:
                    selected.append(eid)
            # 再补充急症（不重复）
            for eid in acute_topk:
                if eid not in selected:
                    selected.append(eid)
            # 4): 补齐逻辑 (已更新)
            if ensure_length:
                target_len = top_k_chronic + top_k_acute
                
                # 阶段一：基于专家关联性的智能补齐
                if len(selected) < target_len and len(selected) > 0:
                    existing_experts = set(selected)
                    padding_candidates = Counter()
                    for eid in existing_experts:
                        for related_eid in range(self.num_experts):
                            if related_eid not in existing_experts:
                                score = self.co_occurrence_matrix[eid, related_eid]
                                if score > 0:
                                    padding_candidates[related_eid] += score
                    for eid, score in padding_candidates.most_common():
                        if len(selected) >= target_len: break
                        if eid not in selected:
                            selected.append(eid)
                
                # 阶段二：使用从矩阵中推断出的热门专家列表进行兜底补齐
                if len(selected) < target_len:
                    for eid in self.popular_experts:
                        if len(selected) >= target_len: break
                        if eid not in selected:
                            selected.append(eid)
        else:
            # 允许重复时，先慢性后急症（长度 = top_k_chronic + top_k_acute）
            selected = list(chronic_topk) + list(acute_topk)
            if ensure_length:
                # 已经保证长度为 top_k_chronic + top_k_acute
                pass

        return selected

# ============================
# 4) 示例（测试）
# ============================
if __name__ == "__main__":
    # 小示例：请用真实的 acute_codes / chronic_codes 集合替换上面占位集合
    # patient_visits：按时间顺序（最早 -> 最新）
    patient_visits_example = [
        ["49", "98"],        # visit 1: diabetes / hypertension (慢性)
        ["127", "139"],      # visit 2: COPD / gastric ulcer (慢性)
        ["122", "2603"],     # visit 3: pneumonia (急症), 交通事故 E-code 模拟为 2603 (急)
    ]

    # 假设我们已经把 acute_codes 和 chronic_codes 完整放入上面的集合
    router = ExpertRouter(
        num_experts=19,
        acute_codes_set=acute_codes,
        chronic_codes_set=chronic_codes,
        recent_k=2  # 仅看最新 2 次 visit 判断急症
    )

    selected = router.select_experts(patient_visits_example, top_k_chronic=2, top_k_acute=2, deduplicate=True, ensure_length=True)
    print("Selected experts:", selected)  # e.g., [ expert_ids... ]

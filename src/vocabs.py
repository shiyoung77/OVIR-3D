from detectron2.data import MetadataCatalog


vocabs = {
    "scannet": [
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
        'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
    ],
    "scannet200": [
        'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed',
        'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
        'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv',
        'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table',
        'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
        'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
        'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
        'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
        'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail',
        'radiator', 'recycling bin', 'container', 'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock',
        'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat',
        'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
        'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
        'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
        'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod',
        'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
        'toilet seat cover dispenser', 'furniture', 'cart', 'storage container', 'scale', 'tissue box',
        'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door',
        'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
        'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse',
        'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand',
        'projector screen', 'divider', 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity',
        'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell',
        'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod', 'coffee kettle', 'structure',
        'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer',
        'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
    ],
    "ycb_video": [
        "blue coffee can", "cracker box", "sugar box", "tomato soup can", "mustard bottle", "tuna fish can",
        "pudding box", "gelatin box", "potted meat can", "banana", "pitcher", "bleach cleanser", "bowl", "mug",
        "power drill", "wood block", "scissors", "large marker", "large clamp", "extra large clamp", "foam",
    ],
    "lvis": MetadataCatalog.get("lvis_v1_val").thing_classes,
}

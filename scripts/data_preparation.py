from sklearn.model_selection import train_test_split

def split_list_file(list_path, train_out, val_out, val_ratio=0.2, seed=42):
    with open(list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    train_lines, val_lines = train_test_split(
        lines, test_size=val_ratio, random_state=seed, shuffle=True
    )

    with open(train_out, "w") as f:
        f.write("\n".join(train_lines))

    with open(val_out, "w") as f:
        f.write("\n".join(val_lines))

    print(f"Train: {len(train_lines)} samples")
    print(f"Val:   {len(val_lines)} samples")


# usage
split_list_file(
    list_path="/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_val_gt.txt",
    train_out="/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/train_gt.txt",
    val_out="/home/student/Dev/LaneNet/data/TUSimple/train_set/seg_label/list/val_gt.txt",
    val_ratio=0.2,   # 20% val
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent $PSScriptRoot
$presentationDir = Join-Path $root 'presentations'
$outPath = Join-Path $presentationDir 'CIFAKE_Final_Presentation_CORRECTED.pptx'
$previewDir = Join-Path $presentationDir 'previews\corrected'
$datasetImage = Join-Path $root 'figures\cifake_real_fake_dataset_samples.png'
$exampleImage = Join-Path $root 'figures\cifake_detection_examples_baseline_tuned_hybrid.png'

New-Item -ItemType Directory -Force -Path $presentationDir, $previewDir | Out-Null

function RGB([int]$r, [int]$g, [int]$b) {
    return [int]($r + (256 * $g) + (65536 * $b))
}

$C = @{
    Navy      = RGB 14 35 62
    Navy2     = RGB 24 52 82
    Ink       = RGB 28 39 55
    Slate     = RGB 88 105 128
    Gray      = RGB 119 136 158
    Light     = RGB 242 246 250
    Line      = RGB 215 224 233
    White     = RGB 255 255 255
    Teal      = RGB 18 154 134
    TealDark  = RGB 12 119 106
    TealLight = RGB 224 247 242
    Blue      = RGB 43 106 176
    BlueLight = RGB 228 239 252
    Orange    = RGB 205 91 47
    OrangeLt  = RGB 252 237 228
    Red       = RGB 195 67 67
    RedLight  = RGB 252 234 234
    Green     = RGB 21 143 99
    GreenLt   = RGB 229 247 239
    Gold      = RGB 234 174 56
    GoldLt    = RGB 255 247 222
}

function Add-Text {
    param(
        $Slide,
        [string]$Text,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [double]$Size = 20,
        [int]$Color = $C.Ink,
        [bool]$Bold = $false,
        [string]$Font = 'Aptos',
        [int]$Align = 1,
        [int]$VAlign = 1
    )
    $shape = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.VerticalAnchor = $VAlign
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = $Font
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Rect {
    param(
        $Slide,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [int]$Fill,
        [int]$Line = -1,
        [double]$LineWeight = 1,
        [bool]$Rounded = $true,
        [double]$Transparency = 0
    )
    $shapeType = if ($Rounded) { 5 } else { 1 }
    $shape = $Slide.Shapes.AddShape($shapeType, $X, $Y, $W, $H)
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $Fill
    $shape.Fill.Transparency = $Transparency
    if ($Line -lt 0) {
        $shape.Line.Visible = 0
    } else {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $Line
        $shape.Line.Weight = $LineWeight
    }
    return $shape
}

function Add-Line {
    param($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [int]$Color = $C.Line, [double]$Weight = 1)
    $line = $Slide.Shapes.AddLine($X1, $Y1, $X2, $Y2)
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = $Weight
    return $line
}

function Add-Pill {
    param($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [int]$Fill, [int]$Color = $C.White, [double]$Size = 12)
    Add-Rect $Slide $X $Y $W 24 $Fill -1 0 $true | Out-Null
    Add-Text $Slide $Text $X ($Y + 2) $W 20 $Size $Color $true 'Aptos' 2 3 | Out-Null
}

function Add-ImageContain {
    param($Slide, [string]$Path, [double]$X, [double]$Y, [double]$W, [double]$H)
    $img = [System.Drawing.Image]::FromFile($Path)
    try {
        $ratio = $img.Width / $img.Height
    } finally {
        $img.Dispose()
    }
    $boxRatio = $W / $H
    if ($ratio -gt $boxRatio) {
        $drawW = $W
        $drawH = $W / $ratio
        $drawX = $X
        $drawY = $Y + (($H - $drawH) / 2)
    } else {
        $drawH = $H
        $drawW = $H * $ratio
        $drawX = $X + (($W - $drawW) / 2)
        $drawY = $Y
    }
    return $Slide.Shapes.AddPicture($Path, 0, -1, $drawX, $drawY, $drawW, $drawH)
}

function Add-Header {
    param($Slide, [string]$Title, [string]$Kicker, [int]$Number)
    Add-Text $Slide $Kicker.ToUpperInvariant() 52 25 520 18 11 $C.TealDark $true | Out-Null
    Add-Text $Slide $Title 52 48 820 48 28 $C.Navy $true 'Aptos Display' | Out-Null
    Add-Line $Slide 52 105 908 105 $C.Line 1 | Out-Null
    Add-Text $Slide ('{0:D2}' -f $Number) 854 28 54 18 10 $C.Gray $true 'Aptos' 3 | Out-Null
}

function Add-Footer {
    param($Slide, [int]$Number, [string]$Source = 'Source: Final.pdf')
    Add-Line $Slide 52 510 908 510 $C.Line 0.8 | Out-Null
    Add-Text $Slide $Source 52 516 760 14 8.5 $C.Gray $false | Out-Null
    Add-Text $Slide ([string]$Number) 854 516 54 14 8.5 $C.Gray $true 'Aptos' 3 | Out-Null
}

function Add-MetricCard {
    param($Slide, [string]$Value, [string]$Label, [double]$X, [double]$Y, [double]$W, [int]$Accent = $C.Teal, [int]$Fill = $C.White)
    Add-Rect $Slide $X $Y $W 88 $Fill $C.Line 0.9 $true | Out-Null
    Add-Rect $Slide $X $Y 6 88 $Accent -1 0 $false | Out-Null
    Add-Text $Slide $Value ($X + 18) ($Y + 12) ($W - 30) 38 27 $Accent $true 'Aptos Display' | Out-Null
    Add-Text $Slide $Label ($X + 18) ($Y + 54) ($W - 30) 22 11.5 $C.Slate $false | Out-Null
}

function Add-BulletBlock {
    param($Slide, [string[]]$Items, [double]$X, [double]$Y, [double]$W, [double]$LineHeight = 46, [double]$Size = 18, [int]$DotColor = $C.Teal, [int]$TextColor = $C.Ink)
    for ($i = 0; $i -lt $Items.Count; $i++) {
        $cy = $Y + ($i * $LineHeight)
        $dot = $Slide.Shapes.AddShape(9, $X, ($cy + 7), 9, 9)
        $dot.Fill.Solid(); $dot.Fill.ForeColor.RGB = $DotColor; $dot.Line.Visible = 0
        Add-Text $Slide $Items[$i] ($X + 20) $cy ($W - 20) ($LineHeight - 2) $Size $TextColor $false | Out-Null
    }
}

function Add-SimpleTable {
    param($Slide, [object[][]]$Data, [double]$X, [double]$Y, [double]$W, [double]$H, [double[]]$ColWidths = $null, [double]$FontSize = 15)
    $rows = $Data.Count
    $cols = $Data[0].Count
    $shape = $Slide.Shapes.AddTable($rows, $cols, $X, $Y, $W, $H)
    $table = $shape.Table
    if ($ColWidths) {
        for ($widthCol = 1; $widthCol -le $cols; $widthCol++) { $table.Columns.Item($widthCol).Width = $ColWidths[$widthCol - 1] }
    }
    for ($row = 1; $row -le $rows; $row++) {
        for ($col = 1; $col -le $cols; $col++) {
            $cell = $table.Cell($row, $col)
            $cell.Shape.TextFrame.TextRange.Text = [string]$Data[$row - 1][$col - 1]
            $cell.Shape.TextFrame.TextRange.Font.Name = 'Aptos'
            $cell.Shape.TextFrame.TextRange.Font.Size = $FontSize
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = if ($row -eq 1) { $C.White } else { $C.Ink }
            $cell.Shape.TextFrame.TextRange.Font.Bold = if ($row -eq 1) { -1 } else { 0 }
            $cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = if ($col -eq 1) { 1 } else { 2 }
            $cell.Shape.TextFrame.MarginLeft = 8
            $cell.Shape.TextFrame.MarginRight = 8
            $cell.Shape.TextFrame.MarginTop = 4
            $cell.Shape.TextFrame.MarginBottom = 2
            $cell.Shape.Fill.Solid()
            if ($row -eq 1) {
                $cell.Shape.Fill.ForeColor.RGB = $C.Navy2
            } elseif (($row % 2) -eq 0) {
                $cell.Shape.Fill.ForeColor.RGB = $C.Light
            } else {
                $cell.Shape.Fill.ForeColor.RGB = $C.White
            }
            $cell.Borders.Item(1).ForeColor.RGB = $C.Line
            $cell.Borders.Item(2).ForeColor.RGB = $C.Line
            $cell.Borders.Item(3).ForeColor.RGB = $C.Line
            $cell.Borders.Item(4).ForeColor.RGB = $C.Line
        }
    }
    return $shape
}

function Add-ConfusionMatrix {
    param($Slide, [double]$X, [double]$Y, [double]$Size, [int]$TN, [int]$FP, [int]$FN, [int]$TP)
    $labelW = 72
    $cell = $Size / 2
    Add-Text $Slide 'PREDICTED' ($X + $labelW) ($Y - 34) $Size 18 10 $C.Gray $true 'Aptos' 2 | Out-Null
    Add-Text $Slide 'REAL' ($X + $labelW) ($Y - 13) $cell 18 11 $C.Slate $true 'Aptos' 2 | Out-Null
    Add-Text $Slide 'FAKE' ($X + $labelW + $cell) ($Y - 13) $cell 18 11 $C.Slate $true 'Aptos' 2 | Out-Null
    Add-Text $Slide 'ACTUAL' ($X - 2) ($Y + $cell - 9) 62 18 10 $C.Gray $true 'Aptos' 2 | Out-Null
    Add-Text $Slide 'REAL' ($X + 2) ($Y + 35) 56 18 11 $C.Slate $true 'Aptos' 3 | Out-Null
    Add-Text $Slide 'FAKE' ($X + 2) ($Y + $cell + 35) 56 18 11 $C.Slate $true 'Aptos' 3 | Out-Null
    $vals = @(
        @($TN, 'True real', $C.GreenLt, $C.Green),
        @($FP, 'Real -> fake', $C.RedLight, $C.Red),
        @($FN, 'Fake -> real', $C.RedLight, $C.Red),
        @($TP, 'True fake', $C.GreenLt, $C.Green)
    )
    for ($i = 0; $i -lt 4; $i++) {
        $row = [math]::Floor($i / 2)
        $col = $i % 2
        $cx = $X + $labelW + ($col * $cell)
        $cy = $Y + ($row * $cell)
        Add-Rect $Slide $cx $cy ($cell - 4) ($cell - 4) $vals[$i][2] $C.White 1 $true | Out-Null
        Add-Text $Slide ([string]$vals[$i][0]) ($cx + 8) ($cy + 22) ($cell - 20) 40 25 $vals[$i][3] $true 'Aptos Display' 2 | Out-Null
        Add-Text $Slide $vals[$i][1] ($cx + 8) ($cy + 62) ($cell - 20) 22 10.5 $C.Slate $false 'Aptos' 2 | Out-Null
    }
}

$ppt = $null
$pres = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = -1
    $ppt.DisplayAlerts = 1
    $pres = $ppt.Presentations.Add()
    $pres.PageSetup.SlideWidth = 960
    $pres.PageSetup.SlideHeight = 540

    # 1. Title
    $s = $pres.Slides.Add(1, 12)
    Add-Rect $s 0 0 960 540 $C.Navy -1 0 $false | Out-Null
    Add-Rect $s 0 0 18 540 $C.Teal -1 0 $false | Out-Null
    Add-Text $s 'HYPERPARAMETER-AWARE EVALUATION' 64 58 520 22 12 $C.Teal $true | Out-Null
    Add-Text $s "Hyperparameter-Aware Evaluation of`nDeep Learning Architectures for`nAI-Generated Image Detection" 64 91 575 149 28 $C.White $true 'Aptos Display' | Out-Null
    Add-Text $s 'CIFAKE · baseline, tuned, ensemble, and diagnostic analysis' 66 251 555 30 15.5 (RGB 185 204 222) $false | Out-Null
    Add-Line $s 66 306 620 306 (RGB 73 101 128) 1 | Out-Null
    Add-Text $s "Ilham Gafarov · Zeyad Qasem · Kerem Düzenli`nBalázs Harangi · Mokhaled N. A. Al-Hamadani" 66 323 575 48 12.5 $C.White $true | Out-Null
    Add-Text $s "University of Debrecen · Debrecen, Hungary`nNorthern Technical University · Kirkuk, Iraq" 66 383 520 38 10.5 (RGB 185 204 222) $false | Out-Null
    Add-Rect $s 661 58 239 392 $C.White -1 0 $true | Out-Null
    $pic = Add-ImageContain $s $datasetImage 681 96 199 230
    $pic.AlternativeText = 'Examples of balanced real CIFAR-10 and synthetic CIFAKE images.'
    Add-Pill $s '120,000 IMAGES' 694 350 173 $C.Teal $C.White 12 | Out-Null
    Add-Text $s 'CIFAKE · REAL vs FAKE' 684 389 194 22 11 $C.Navy $true 'Aptos' 2 | Out-Null
    Add-Text $s 'Final research presentation' 66 484 340 16 10 (RGB 148 171 194) $false | Out-Null

    # 2. Problem and question
    $s = $pres.Slides.Add(2, 12); Add-Header $s 'Why this problem matters' 'Research context' 2
    Add-Rect $s 52 131 396 326 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'Synthetic images are becoming easier to create — and harder to authenticate.' 78 159 344 118 27 $C.White $true 'Aptos Display' | Out-Null
    Add-Text $s 'Journalism · law enforcement · social media · digital forensics' 78 299 328 65 15 (RGB 200 216 231) $false | Out-Null
    Add-Pill $s 'THE GAP' 78 391 86 $C.Orange $C.White 11 | Out-Null
    Add-Text $s 'Few controlled studies compare how tuning choices interact with different detector architectures.' 176 389 240 58 14 $C.White $false | Out-Null
    Add-Text $s 'Research question' 491 143 350 24 12 $C.TealDark $true | Out-Null
    Add-Text $s 'How much does hyperparameter optimization change AI-image detection performance across CNNs, transformers, and ensembles?' 491 177 401 124 25 $C.Navy $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('Compare four pretrained architectures', 'Measure baseline versus tuned performance', 'Test whether soft voting adds complementary value') 496 330 385 48 16 $C.Teal
    Add-Footer $s 2

    # 3. Study design
    $s = $pres.Slides.Add(3, 12); Add-Header $s 'Study design at a glance' 'Scope' 3
    $cards = @(
        @('120k', 'balanced images', $C.Blue, $C.BlueLight),
        @('4', 'model architectures', $C.TealDark, $C.TealLight),
        @('2', 'training conditions', $C.Orange, $C.OrangeLt),
        @('20k', 'held-out test images', $C.Green, $C.GreenLt)
    )
    for ($i = 0; $i -lt 4; $i++) {
        $x = 52 + ($i * 218)
        Add-Rect $s $x 132 196 105 $cards[$i][3] -1 0 $true | Out-Null
        Add-Text $s $cards[$i][0] ($x + 16) 145 164 44 31 $cards[$i][2] $true 'Aptos Display' | Out-Null
        Add-Text $s $cards[$i][1] ($x + 16) 194 164 20 12 $C.Slate $false | Out-Null
    }
    Add-Text $s 'One controlled comparison' 52 275 260 24 13 $C.TealDark $true | Out-Null
    $steps = @('Pretrained backbones', 'Baseline training', 'Configuration search', 'Full retraining', 'Official test')
    $stepColors = @($C.Navy2, $C.Slate, $C.Orange, $C.TealDark, $C.Green)
    for ($i = 0; $i -lt $steps.Count; $i++) {
        $x = 52 + ($i * 171)
        Add-Rect $s $x 317 143 72 $C.White $C.Line 1 $true | Out-Null
        Add-Text $s ([string]($i + 1)) ($x + 12) 331 24 24 14 $stepColors[$i] $true 'Aptos' 2 | Out-Null
        Add-Text $s $steps[$i] ($x + 38) 327 94 44 13 $C.Ink $true 'Aptos' 1 3 | Out-Null
        if ($i -lt 4) {
            Add-Line $s ($x + 143) 353 ($x + 169) 353 $C.Line 2 | Out-Null
        }
    }
    Add-Rect $s 52 417 856 62 $C.Light -1 0 $true | Out-Null
    Add-Text $s 'Primary metric' 71 431 110 15 10.5 $C.Gray $true | Out-Null
    Add-Text $s 'Accuracy' 71 449 110 20 14 $C.Navy $true | Out-Null
    Add-Text $s 'Secondary metrics' 224 431 130 15 10.5 $C.Gray $true | Out-Null
    Add-Text $s 'Precision · Recall · F1' 224 449 190 20 14 $C.Navy $true | Out-Null
    Add-Text $s 'Positive class' 470 431 110 15 10.5 $C.Gray $true | Out-Null
    Add-Text $s 'FAKE' 470 449 90 20 14 $C.Orange $true | Out-Null
    Add-Text $s 'Decision rule' 631 431 110 15 10.5 $C.Gray $true | Out-Null
    Add-Text $s 'Best validation-weighted F1' 631 449 230 20 13 $C.Navy $true | Out-Null
    Add-Footer $s 3

    # 4. Dataset
    $s = $pres.Slides.Add(4, 12); Add-Header $s 'CIFAKE: balanced real and synthetic imagery' 'Dataset & preprocessing' 4
    Add-Rect $s 52 128 504 340 $C.White $C.Line 1 $true | Out-Null
    $pic = Add-ImageContain $s $datasetImage 68 145 472 306
    $pic.AlternativeText = 'Sample CIFAKE images: five real CIFAR-10 images and five synthetic images.'
    Add-MetricCard $s '100,000' 'training images · 50/50 split' 590 135 318 $C.Blue $C.BlueLight
    Add-MetricCard $s '20,000' 'official test images · untouched during tuning' 590 240 318 $C.TealDark $C.TealLight
    Add-Rect $s 590 345 318 118 $C.Light -1 0 $true | Out-Null
    Add-Text $s 'PREPROCESSING' 609 361 220 18 10.5 $C.Gray $true | Out-Null
    Add-Text $s 'ImageNet normalization' 609 387 260 22 15 $C.Navy $true | Out-Null
    Add-Text $s "224×224 for all four architectures`nResizing does not add source detail" 609 416 270 38 12.5 $C.Slate $false | Out-Null
    Add-Footer $s 4 'Source: Final.pdf; Bird & Lotfi (2024), CIFAKE'

    # 5. Architectures
    $s = $pres.Slides.Add(5, 12); Add-Header $s 'Four complementary architectures' 'Model families' 5
    $modelCards = @(
        @('R', 'ResNet-50', 'Residual CNN', 'Skip connections support deep feature learning.', $C.Blue, $C.BlueLight),
        @('E', 'EfficientNetV2-S', 'Efficient CNN', 'Fused-MBConv design targets speed and parameter efficiency.', $C.TealDark, $C.TealLight),
        @('V', 'ViT-B/16', 'Vision transformer', '16×16 patches and self-attention capture global relationships.', $C.Orange, $C.OrangeLt),
        @('X', 'Xception', 'Separable CNN', 'Depthwise separable convolutions target forensic artifacts.', $C.Green, $C.GreenLt)
    )
    for ($i = 0; $i -lt 4; $i++) {
        $x = 52 + (($i % 2) * 433)
        $y = 132 + ([math]::Floor($i / 2) * 171)
        Add-Rect $s $x $y 410 145 $C.White $C.Line 1 $true | Out-Null
        Add-Rect $s ($x + 16) ($y + 18) 58 58 $modelCards[$i][5] -1 0 $true | Out-Null
        Add-Text $s $modelCards[$i][0] ($x + 16) ($y + 27) 58 38 24 $modelCards[$i][4] $true 'Aptos Display' 2 3 | Out-Null
        Add-Text $s $modelCards[$i][1] ($x + 92) ($y + 16) 292 28 20 $C.Navy $true 'Aptos Display' | Out-Null
        Add-Text $s $modelCards[$i][2].ToUpperInvariant() ($x + 92) ($y + 48) 290 17 9.5 $modelCards[$i][4] $true | Out-Null
        Add-Text $s $modelCards[$i][3] ($x + 92) ($y + 76) 286 52 12.5 $C.Slate $false | Out-Null
    }
    Add-Text $s 'All models used ImageNet-pretrained weights and a two-class output head.' 52 479 700 20 12 $C.Gray $false | Out-Null
    Add-Footer $s 5

    # 6. Pipeline
    $s = $pres.Slides.Add(6, 12); Add-Header $s 'Experimental pipeline' 'Method' 6
    $pipeline = @(
        @('1', 'INGEST', "Balanced CIFAKE`ntrain + test"),
        @('2', 'PREPARE', "Resize, normalize,`naugment"),
        @('3', 'BASELINE', "10 epochs on`n100k images"),
        @('4', 'SEARCH', "4k train +`n1k validation"),
        @('5', 'RETRAIN', "Best config on`nfull training split"),
        @('6', 'EVALUATE', "20k official`ntest images")
    )
    for ($i = 0; $i -lt 6; $i++) {
        $x = 43 + ($i * 151)
        $fill = if ($i -eq 3) { $C.OrangeLt } elseif ($i -eq 5) { $C.TealLight } else { $C.Light }
        $accent = if ($i -eq 3) { $C.Orange } elseif ($i -eq 5) { $C.TealDark } else { $C.Navy2 }
        Add-Rect $s $x 161 126 185 $fill $C.Line 1 $true | Out-Null
        Add-Pill $s $pipeline[$i][0] ($x + 44) 177 38 $accent $C.White 12 | Out-Null
        Add-Text $s $pipeline[$i][1] ($x + 10) 219 106 20 10.5 $accent $true 'Aptos' 2 | Out-Null
        Add-Text $s $pipeline[$i][2] ($x + 12) 257 102 53 13 $C.Ink $false 'Aptos' 2 | Out-Null
        if ($i -lt 5) { Add-Line $s ($x + 126) 253 ($x + 147) 253 $C.Gray 2 | Out-Null }
    }
    Add-Rect $s 53 383 854 82 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'Evaluation safeguards' 76 398 170 18 11 (RGB 94 216 194) $true | Out-Null
    Add-Text $s 'Deep models: reserved 20,000-image official test set' 76 423 390 22 14.5 $C.White $true | Out-Null
    Add-Text $s 'Diagnostics: disjoint balanced 4,000-image test subset' 486 423 390 22 14.5 $C.White $true | Out-Null
    Add-Footer $s 6

    # 7. Baseline vs tuning
    $s = $pres.Slides.Add(7, 12); Add-Header $s 'Baseline and tuned conditions' 'Controlled comparison' 7
    Add-Rect $s 52 135 404 327 $C.Light -1 0 $true | Out-Null
    Add-Pill $s 'BASELINE' 77 156 102 $C.Slate $C.White 11 | Out-Null
    Add-Text $s 'Common setup' 77 199 315 28 21 $C.Navy $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('100,000 training images', '10 epochs · batch size 16', 'AdamW · learning rate 1e-4', 'Weight decay 1e-4', 'Random horizontal flip') 82 243 334 38 14.5 $C.Slate
    Add-Rect $s 484 135 424 327 $C.TealLight -1 0 $true | Out-Null
    Add-Pill $s 'TUNED' 509 156 88 $C.TealDark $C.White 11 | Out-Null
    Add-Text $s 'Controlled search + retraining' 509 199 360 28 21 $C.Navy $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('4,000-image trial training subset', '1,000-image validation subset', '3 epochs per candidate configuration', 'Select by validation-weighted F1', 'Retrain selected setting for 10 epochs') 514 243 354 38 14.5 $C.TealDark
    Add-Footer $s 7

    # 8. Search space
    $s = $pres.Slides.Add(8, 12); Add-Header $s 'Hyperparameter search space' 'Tuning design' 8
    $searchData = @(
        @('Hyperparameter', 'Candidate values'),
        @('Learning rate', '1e-3 · 1e-4 · 5e-5 · 1e-5'),
        @('Batch size', '8 · 16 · 32'),
        @('Optimizer', 'AdamW'),
        @('Tuned input size', '224×224'),
        @('Weight decay', '0 · 1e-4'),
        @('Augmentation', 'On · Off'),
        @('Freeze backbone', 'True · False'),
        @('Trial epochs', '3')
    )
    Add-SimpleTable $s $searchData 52 133 548 338 @(210,338) 14.5 | Out-Null
    Add-Rect $s 630 133 278 338 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'What the search isolates' 655 158 220 28 18 $C.White $true 'Aptos Display' | Out-Null
    Add-Line $s 655 198 879 198 (RGB 69 99 128) 1 | Out-Null
    Add-Text $s '01' 655 220 30 24 13 $C.Teal $true | Out-Null
    Add-Text $s 'Optimization scale' 697 218 174 24 15 $C.White $true | Out-Null
    Add-Text $s 'How aggressively pretrained weights are updated.' 697 246 176 48 12.5 (RGB 196 214 230) $false | Out-Null
    Add-Text $s '02' 655 312 30 24 13 $C.Teal $true | Out-Null
    Add-Text $s 'Feature adaptation' 697 310 174 24 15 $C.White $true | Out-Null
    Add-Text $s 'Whether the backbone learns CIFAKE-specific evidence.' 697 338 176 48 12.5 (RGB 196 214 230) $false | Out-Null
    Add-Text $s 'Selection: validation-weighted F1' 655 426 225 20 12 (RGB 94 216 194) $true | Out-Null
    Add-Footer $s 8

    # 9. Evaluation
    $s = $pres.Slides.Add(9, 12); Add-Header $s 'Evaluation framework' 'Metrics' 9
    $metricDefs = @(
        @('ACC', 'Accuracy', 'Overall share of correct predictions', $C.Blue, $C.BlueLight),
        @('P', 'Precision', 'When FAKE is predicted, how often it is correct', $C.TealDark, $C.TealLight),
        @('R', 'Recall', 'Share of FAKE images successfully detected', $C.Orange, $C.OrangeLt),
        @('F1', 'F1-score', 'Balance between fake-class precision and recall', $C.Green, $C.GreenLt)
    )
    for ($i = 0; $i -lt 4; $i++) {
        $x = 52 + ($i * 218)
        Add-Rect $s $x 138 196 234 $C.White $C.Line 1 $true | Out-Null
        Add-Rect $s ($x + 20) 160 56 56 $metricDefs[$i][4] -1 0 $true | Out-Null
        Add-Text $s $metricDefs[$i][0] ($x + 20) 173 56 28 17 $metricDefs[$i][3] $true 'Aptos Display' 2 | Out-Null
        Add-Text $s $metricDefs[$i][1] ($x + 20) 237 156 28 18 $C.Navy $true 'Aptos Display' | Out-Null
        Add-Text $s $metricDefs[$i][2] ($x + 20) 277 156 70 12.5 $C.Slate $false | Out-Null
    }
    Add-Rect $s 52 403 856 71 $C.Light -1 0 $true | Out-Null
    Add-Pill $s 'FAKE = POSITIVE CLASS' 74 427 175 $C.Orange $C.White 10.5 | Out-Null
    Add-Text $s 'Confusion matrices expose the remaining false positives and false negatives.' 275 424 588 28 16 $C.Navy $true | Out-Null
    Add-Footer $s 9

    # 10. Main accuracy comparison
    $s = $pres.Slides.Add(10, 12); Add-Header $s 'Hyperparameter tuning improved every model' 'Headline result' 10
    $models = @('ResNet-50', 'EfficientNetV2-S', 'ViT-B/16', 'Xception', 'Hybrid')
    $base = @(98.13, 75.67, 95.95, 97.74, 98.55)
    $tuned = @(98.26, 97.98, 98.95, 98.17, 99.07)
    $gains = @('+0.13 pp', '+22.31 pp', '+3.00 pp', '+0.43 pp', '+0.52 pp')
    $chartX = 76; $chartY = 156; $chartW = 808; $chartH = 270; $minY = 70; $maxY = 100
    Add-Text $s 'Accuracy (%) · scale 70–100' 52 116 190 18 10 $C.Gray $true | Out-Null
    for ($v = 70; $v -le 100; $v += 5) {
        $yy = $chartY + $chartH - (($v - $minY) / ($maxY - $minY) * $chartH)
        Add-Line $s $chartX $yy ($chartX + $chartW) $yy $C.Line 0.8 | Out-Null
        Add-Text $s ([string]$v) 42 ($yy - 8) 28 16 9.5 $C.Gray $false 'Aptos' 3 | Out-Null
    }
    for ($i = 0; $i -lt 5; $i++) {
        $groupX = $chartX + 35 + ($i * 158)
        $bh = ($base[$i] - $minY) / ($maxY - $minY) * $chartH
        $th = ($tuned[$i] - $minY) / ($maxY - $minY) * $chartH
        Add-Rect $s $groupX ($chartY + $chartH - $bh) 43 $bh $C.Slate -1 0 $false | Out-Null
        $tColor = if ($i -eq 4) { $C.Green } else { $C.Teal }
        Add-Rect $s ($groupX + 49) ($chartY + $chartH - $th) 43 $th $tColor -1 0 $false | Out-Null
        Add-Text $s ('{0:N2}' -f $base[$i]) ($groupX - 6) ($chartY + $chartH - $bh - 21) 55 18 9.8 $C.Slate $true 'Aptos' 2 | Out-Null
        Add-Text $s ('{0:N2}' -f $tuned[$i]) ($groupX + 43) ($chartY + $chartH - $th - 21) 60 18 9.8 $tColor $true 'Aptos' 2 | Out-Null
        Add-Text $s $models[$i] ($groupX - 16) 433 124 22 10.5 $C.Ink $true 'Aptos' 2 | Out-Null
        Add-Text $s $gains[$i] ($groupX - 7) 458 106 18 10 $C.Orange $true 'Aptos' 2 | Out-Null
    }
    Add-Rect $s 667 115 12 12 $C.Slate -1 0 $false | Out-Null
    Add-Text $s 'Baseline' 685 112 73 18 10 $C.Slate $true | Out-Null
    Add-Rect $s 770 115 12 12 $C.Teal -1 0 $false | Out-Null
    Add-Text $s 'Tuned' 788 112 60 18 10 $C.TealDark $true | Out-Null
    Add-Footer $s 10 'Source: Final.pdf, Table II (official 20,000-image test split)'

    # 11. Results matrix
    $s = $pres.Slides.Add(11, 12); Add-Header $s 'Complete test-set results' 'Performance matrix' 11
    $results = @(
        @('Model / condition', 'Acc.', 'Prec.', 'Recall', 'F1'),
        @('Baseline · ResNet-50', '98.13', '98.21', '98.04', '98.12'),
        @('Baseline · EfficientNetV2-S', '75.67', '69.76', '90.60', '78.83'),
        @('Baseline · ViT-B/16', '95.95', '94.41', '97.67', '96.01'),
        @('Baseline · Xception', '97.74', '97.81', '97.68', '97.74'),
        @('Tuned · ResNet-50', '98.26', '97.94', '98.59', '98.27'),
        @('Tuned · EfficientNetV2-S', '97.98', '98.98', '96.95', '97.95'),
        @('Tuned · ViT-B/16', '98.95', '98.71', '99.19', '98.95'),
        @('Tuned · Xception', '98.17', '97.26', '99.13', '98.19'),
        @('Baseline Hybrid', '98.55', '98.36', '98.75', '98.55'),
        @('Tuned Hybrid', '99.07', '99.01', '99.13', '99.07')
    )
    $tbl = Add-SimpleTable $s $results 52 128 856 352 @(350,126.5,126.5,126.5,126.5) 13.5
    for ($col = 1; $col -le 5; $col++) {
        $tbl.Table.Cell(11, $col).Shape.Fill.ForeColor.RGB = $C.TealLight
        $tbl.Table.Cell(11, $col).Shape.TextFrame.TextRange.Font.Bold = -1
        $tbl.Table.Cell(11, $col).Shape.TextFrame.TextRange.Font.Color.RGB = if ($col -eq 1) { $C.Navy } else { $C.TealDark }
    }
    Add-Text $s 'All values are percentages; precision, recall, and F1 use FAKE as the positive class.' 52 487 700 14 9.5 $C.Gray $false | Out-Null
    Add-Footer $s 11 'Source: Final.pdf, Table II'

    # 12. EfficientNet spotlight
    $s = $pres.Slides.Add(12, 12); Add-Header $s 'The largest gain came from EfficientNetV2-S' 'Hyperparameter sensitivity' 12
    Add-Rect $s 52 137 405 329 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s '75.67%' 78 166 250 60 42 $C.White $true 'Aptos Display' | Out-Null
    Add-Text $s 'baseline accuracy' 80 224 230 22 13 (RGB 190 209 227) $false | Out-Null
    Add-Text $s '→' 80 269 70 50 34 $C.Teal $true 'Aptos Display' | Out-Null
    Add-Text $s '97.98%' 153 266 230 60 42 $C.Teal $true 'Aptos Display' | Out-Null
    Add-Text $s 'tuned accuracy' 155 324 230 22 13 (RGB 190 209 227) $false | Out-Null
    Add-Pill $s '+22.31 PERCENTAGE POINTS' 79 390 282 $C.Orange $C.White 11 | Out-Null
    Add-Text $s 'Why?' 497 151 100 24 12 $C.TealDark $true | Out-Null
    Add-Text $s 'A frozen backbone could not adapt sufficiently to the domain shift from ImageNet to low-resolution CIFAKE imagery.' 497 183 385 90 22 $C.Navy $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('Unfreeze the backbone', 'Use AdamW with 1e-4 weight decay', 'Use a 1e-4 learning rate') 501 308 360 48 16 $C.Teal
    Add-Footer $s 12

    # 13. ViT
    $s = $pres.Slides.Add(13, 12); Add-Header $s 'ViT-B/16 was the best individual detector' 'Best single model' 13
    Add-MetricCard $s '98.95%' 'accuracy' 52 137 194 $C.Orange $C.OrangeLt
    Add-MetricCard $s '98.71%' 'fake precision' 264 137 194 $C.TealDark $C.TealLight
    Add-MetricCard $s '99.19%' 'fake recall' 476 137 194 $C.Green $C.GreenLt
    Add-MetricCard $s '98.95%' 'fake F1' 688 137 194 $C.Blue $C.BlueLight
    Add-Rect $s 52 257 404 211 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'Selected configuration' 78 280 300 26 18 $C.White $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('Learning rate: 1e-5', 'Weight decay: 1e-4', 'AdamW + augmentation', 'Backbone fully trainable') 82 324 320 34 14 (RGB 94 216 194) $C.White
    Add-ConfusionMatrix $s 520 280 210 9870 130 81 9919
    Add-Text $s '211 errors / 20,000 images' 575 490 275 16 10.5 $C.Slate $true 'Aptos' 2 | Out-Null
    Add-Footer $s 13 'Source: Final.pdf and saved ViT-B/16 test artifact'

    # 14. Ensemble
    $s = $pres.Slides.Add(14, 12); Add-Header $s 'Soft voting produced the strongest result' 'Tuned hybrid ensemble' 14
    Add-Rect $s 52 137 358 330 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'Weighted probability fusion' 77 161 305 28 18 $C.White $true 'Aptos Display' | Out-Null
    Add-Text $s 'p̂ = Σ wᵢ · pᵢ' 77 218 250 44 30 (RGB 94 216 194) $true 'Aptos Display' | Out-Null
    Add-Text $s 'Σ wᵢ = 1; wᵢ ≥ 0' 78 270 250 22 14 (RGB 190 209 227) $false | Out-Null
    Add-Line $s 78 315 381 315 (RGB 68 98 127) 1 | Out-Null
    Add-Text $s 'Selected weights' 78 337 150 18 12 $C.TealLight $true | Out-Null
    Add-Text $s 'ResNet-50' 78 373 110 18 11.5 $C.White $false | Out-Null
    Add-Text $s '0%' 336 373 40 18 11.5 $C.White $true 'Aptos' 3 | Out-Null
    $names = @('EfficientNetV2-S', 'ViT-B/16', 'Xception')
    for ($i = 0; $i -lt 3; $i++) {
        $yy = 399 + ($i * 24)
        Add-Text $s $names[$i] 78 $yy 145 18 11.5 $C.White $false | Out-Null
        Add-Rect $s 225 ($yy + 3) 100 10 (RGB 61 91 120) -1 0 $false | Out-Null
        Add-Rect $s 225 ($yy + 3) 100 10 $C.Teal -1 0 $false | Out-Null
        Add-Text $s '33.3%' 336 $yy 40 18 11.5 $C.White $true 'Aptos' 3 | Out-Null
    }
    Add-Text $s '99.07%' 474 159 355 72 51 $C.TealDark $true 'Aptos Display' | Out-Null
    Add-Text $s 'official test accuracy' 477 231 330 24 16 $C.Slate $false | Out-Null
    Add-MetricCard $s '99.01%' 'fake precision' 476 283 196 $C.TealDark $C.TealLight
    Add-MetricCard $s '99.13%' 'fake recall' 690 283 196 $C.Green $C.GreenLt
    Add-Rect $s 476 394 410 72 $C.Light -1 0 $true | Out-Null
    Add-Text $s '+0.12 pp vs ViT-B/16' 496 409 370 24 18 $C.Navy $true | Out-Null
    Add-Text $s '+0.52 pp vs baseline hybrid' 496 438 370 18 11.5 $C.Orange $true | Out-Null
    Add-Footer $s 14

    # 15. Error profile
    $s = $pres.Slides.Add(15, 12); Add-Header $s 'Only 186 test images were misclassified' 'Tuned ensemble error profile' 15
    Add-ConfusionMatrix $s 92 184 250 9901 99 87 9913
    Add-Rect $s 526 139 382 329 $C.Light -1 0 $true | Out-Null
    Add-Text $s 'Error balance' 551 162 300 26 19 $C.Navy $true 'Aptos Display' | Out-Null
    Add-MetricCard $s '99' 'real images predicted as fake' 551 207 332 $C.Red $C.RedLight
    Add-MetricCard $s '87' 'fake images predicted as real' 551 312 332 $C.Orange $C.OrangeLt
    Add-Text $s '0.93% total error rate' 551 425 250 22 14 $C.TealDark $true | Out-Null
    Add-Footer $s 15 'Source: Final.pdf, tuned hybrid confusion matrix'

    # 16. Qualitative example
    $s = $pres.Slides.Add(16, 12); Add-Header $s 'Tuning corrected high-confidence mistakes' 'Qualitative examples' 16
    Add-Rect $s 52 131 856 315 $C.White $C.Line 1 $true | Out-Null
    $pic = Add-ImageContain $s $exampleImage 65 144 830 288
    $pic.AlternativeText = 'Two CIFAKE examples misclassified by baseline ViT-B/16 and corrected by tuned ViT-B/16 and the tuned hybrid ensemble.'
    Add-Rect $s 52 458 856 38 $C.TealLight -1 0 $true | Out-Null
    Add-Text $s 'Both examples were misclassified by baseline ViT-B/16, then correctly classified by the tuned ViT and ensemble.' 71 468 816 18 12.5 $C.TealDark $true 'Aptos' 2 | Out-Null
    Add-Footer $s 16 'Source: Final project figure; illustrative examples from CIFAKE'

    # 17. Low-level diagnostic
    $s = $pres.Slides.Add(17, 12); Add-Header $s 'Frequency cues help — but do not explain deep models' 'Low-level diagnostic' 17
    Add-Rect $s 52 133 410 338 $C.Light -1 0 $true | Out-Null
    Add-Text $s 'Statistical representations' 76 155 330 26 18 $C.Navy $true 'Aptos Display' | Out-Null
    Add-Rect $s 76 198 360 104 $C.White $C.Line 1 $true | Out-Null
    Add-Text $s 'RGB histograms' 94 214 220 24 15 $C.Navy $true | Out-Null
    Add-Text $s '66.45–67.40%' 94 243 220 36 25 $C.Orange $true 'Aptos Display' | Out-Null
    Add-Text $s 'accuracy · AUC 73.19–73.67%' 94 278 290 18 10.5 $C.Slate $false | Out-Null
    Add-Rect $s 76 321 360 118 $C.White $C.Line 1 $true | Out-Null
    Add-Text $s 'Fourier radial log spectrum' 94 337 285 24 15 $C.Navy $true | Out-Null
    Add-Text $s '80.70–80.75%' 94 366 220 36 25 $C.TealDark $true 'Aptos Display' | Out-Null
    Add-Text $s 'accuracy · AUC 87.49–87.54%' 94 405 290 18 10.5 $C.Slate $false | Out-Null
    Add-Rect $s 484 133 424 338 $C.Navy2 -1 0 $true | Out-Null
    Add-Text $s 'Learned predictions · same 4,000 images' 510 155 360 26 18 $C.White $true 'Aptos Display' | Out-Null
    Add-Rect $s 510 201 176 133 $C.TealLight -1 0 $true | Out-Null
    Add-Text $s 'Tuned ViT-B/16' 526 218 145 20 12 $C.Navy $true | Out-Null
    Add-Text $s '98.72%' 526 253 145 40 28 $C.TealDark $true 'Aptos Display' | Out-Null
    Add-Text $s 'AUC 99.91%' 526 302 130 18 10.5 $C.Slate $false | Out-Null
    Add-Rect $s 708 201 176 133 $C.GreenLt -1 0 $true | Out-Null
    Add-Text $s 'Tuned hybrid' 724 218 145 20 12 $C.Navy $true | Out-Null
    Add-Text $s '98.78%' 724 253 145 40 28 $C.Green $true 'Aptos Display' | Out-Null
    Add-Text $s 'AUC 99.92%' 724 302 130 18 10.5 $C.Slate $false | Out-Null
    Add-Line $s 510 362 884 362 (RGB 70 101 130) 1 | Out-Null
    Add-Text $s 'Interpretation' 510 382 120 18 11.5 $C.TealLight $true | Out-Null
    Add-Text $s 'Frequency artifacts are informative, but richer learned evidence accounts for substantially more of the classification performance.' 510 407 360 50 13 $C.White $false | Out-Null
    Add-Footer $s 17 'Source: Final.pdf, Table IV (balanced 4,000-image test subset)'

    # 18. Recommendations and limitations
    $s = $pres.Slides.Add(18, 12); Add-Header $s 'What practitioners should do — and verify' 'Implications' 18
    Add-Rect $s 52 135 407 336 $C.TealLight -1 0 $true | Out-Null
    Add-Text $s 'Recommended' 77 158 260 26 19 $C.TealDark $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('Use tuned ViT-B/16 for the strongest accuracy–efficiency trade-off.', 'Use the ensemble for offline or server-side maximum accuracy.', 'Unfreeze EfficientNet and tune learning rate with weight decay.', 'Treat the +0.12 pp ensemble margin cautiously.') 82 211 335 59 14.5 $C.TealDark
    Add-Rect $s 485 135 423 336 $C.OrangeLt -1 0 $true | Out-Null
    Add-Text $s 'Limitations to communicate' 510 158 340 26 19 $C.Orange $true 'Aptos Display' | Out-Null
    Add-BulletBlock $s @('ImageNet initialization confounds architecture-only comparisons.', 'CIFAKE is low-resolution and uses one dataset/generative setting.', 'Results use single training runs without seeds, confidence intervals, or significance tests.', 'Compression and unseen-generator robustness remain untested.') 515 211 350 59 14.5 $C.Orange
    Add-Footer $s 18

    # 19. Conclusions
    $s = $pres.Slides.Add(19, 12); Add-Header $s 'Three takeaways' 'Conclusion' 19
    $takeaways = @(
        @('01', 'Tuning is architecture-dependent', 'The effect ranged from +0.13 pp for ResNet-50 to +22.31 pp for EfficientNetV2-S.', $C.Blue, $C.BlueLight),
        @('02', 'ViT leads; spectral cues are incomplete', 'Tuned ViT-B/16 reached 98.95%; Fourier baselines reached about 80.75% on the diagnostic subset.', $C.Orange, $C.OrangeLt),
        @('03', 'The ensemble leads — by a small margin', 'The tuned hybrid reached 99.07%, only +0.12 pp above ViT-B/16, with added computational cost.', $C.TealDark, $C.TealLight)
    )
    for ($i = 0; $i -lt 3; $i++) {
        $y = 137 + ($i * 111)
        Add-Rect $s 52 $y 856 91 $takeaways[$i][4] -1 0 $true | Out-Null
        Add-Text $s $takeaways[$i][0] 76 ($y + 23) 60 32 21 $takeaways[$i][3] $true 'Aptos Display' | Out-Null
        Add-Text $s $takeaways[$i][1] 151 ($y + 14) 420 28 18.5 $C.Navy $true 'Aptos Display' | Out-Null
        Add-Text $s $takeaways[$i][2] 151 ($y + 48) 710 30 13 $C.Slate $false | Out-Null
    }
    Add-Text $s 'Next: multi-seed statistical tests · high-resolution, cross-generator data · semantic + frequency multi-branch models' 52 477 856 20 12 $C.Navy $true 'Aptos' 2 | Out-Null
    Add-Footer $s 19

    # 20. References
    $s = $pres.Slides.Add(20, 12); Add-Header $s 'Selected references' 'Sources' 20
    $refsLeft = @(
        'Bird, J. J., & Lotfi, A. (2024). CIFAKE: Image classification and explainable identification of AI-generated synthetic images. IEEE Access, 12, 15642–15650.',
        'He, K. et al. (2016). Deep residual learning for image recognition. CVPR.',
        'Dosovitskiy, A. et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. ICLR.',
        'Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. ICML.'
    )
    $refsRight = @(
        'Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. CVPR.',
        'Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. ICLR.',
        'Wang, S.-Y. et al. (2020). CNN-generated images are surprisingly easy to spot … for now. CVPR.',
        'Primary source: Final.pdf — Hyperparameter-Aware Evaluation of Deep Learning Architectures for AI-Generated Image Detection.'
    )
    Add-Rect $s 52 133 410 334 $C.Light -1 0 $true | Out-Null
    Add-Rect $s 484 133 424 334 $C.Light -1 0 $true | Out-Null
    for ($i = 0; $i -lt 4; $i++) {
        $yy = 153 + ($i * 76)
        Add-Text $s ([string]($i + 1)) 74 $yy 24 22 11 $C.TealDark $true | Out-Null
        Add-Text $s $refsLeft[$i] 105 $yy 332 66 11.5 $C.Ink $false | Out-Null
        Add-Text $s ([string]($i + 5)) 506 $yy 24 22 11 $C.TealDark $true | Out-Null
        Add-Text $s $refsRight[$i] 537 $yy 342 66 11.5 $C.Ink $false | Out-Null
    }
    Add-Text $s 'Full bibliography: see Final.pdf.' 52 484 856 18 11 $C.Gray $false 'Aptos' 2 | Out-Null
    Add-Footer $s 20 'Source: bibliography in Final.pdf'

    $pres.SaveAs($outPath, 24)
    if (-not (Test-Path -LiteralPath $previewDir)) {
        New-Item -ItemType Directory -Path $previewDir | Out-Null
    }
    $pres.Export($previewDir, 'PNG', 1600, 900)
    $slideCount = $pres.Slides.Count
    $pres.Close()
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($pres) | Out-Null
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($ppt) | Out-Null
    Write-Output "Created: $outPath"
    Write-Output "Slides: $slideCount"
    Write-Output "Preview: $previewDir"
} catch {
    if ($pres) { try { $pres.Close() } catch {} }
    if ($ppt) { try { $ppt.Quit() } catch {} }
    throw
}

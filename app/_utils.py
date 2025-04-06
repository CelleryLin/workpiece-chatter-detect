def add_slider(parent_layout, label_text, min_val, max_val, default_val, tick_interval=1, decimals=0):
    label = QLabel(label_text)
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default_val)
    slider.setTickPosition(QSlider.TicksBelow)
    slider.setTickInterval(tick_interval)
    slider.valueChanged.connect(self.auto_process)
    
    # Display current value
    value_label = QLabel(f"Value: {default_val if decimals == 0 else default_val/10**decimals:.1f}")
    
    def update_label(value):
        display_value = value if decimals == 0 else value/10**decimals
        value_label.setText(f"Value: {display_value:.{decimals}f}")
    
    slider.valueChanged.connect(update_label)
    
    parent_layout.addWidget(label)
    parent_layout.addWidget(slider)
    parent_layout.addWidget(value_label)
    return slider
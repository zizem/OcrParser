package org.example.clipBoard;

import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;

//Клас для збереження фотографії у буфер обміну так як звичайний буфер обміну НЕ ПІДТРИМУЄ img
public class  ImageSelectWrapper implements Transferable {

    private final Image image;

    public ImageSelectWrapper(Image image) {
        this.image = image;
    }
//Повертає список форматів
    @Override
    public DataFlavor[] getTransferDataFlavors() {
        return new DataFlavor[]{
                DataFlavor.imageFlavor
        };
    }
//Перевірка чи підтримується формат
    @Override
    public boolean isDataFlavorSupported(DataFlavor flavor) {
        return DataFlavor.imageFlavor.equals(flavor);
    }
//повертає тип данних для зображення
    @Override
    public Object getTransferData(DataFlavor flavor) throws UnsupportedFlavorException {
        if (!DataFlavor.imageFlavor.equals(flavor)) {
            throw new UnsupportedFlavorException(flavor);
        }
        return image;
    }
}
